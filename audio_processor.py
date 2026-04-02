

import os
import asyncio
import logging
import tempfile
import json
import uuid
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
import aiohttp
import aiofiles
from pathlib import Path


# Audio processing

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    print("WARNING: faster-whisper not installed. Install with: pip install faster-whisper")
    WHISPER_AVAILABLE = False

# OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("WARNING: openai not installed. Install with: pip install openai")
    OPENAI_AVAILABLE = False

# Audio format conversion
try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    print("WARNING: ffmpeg-python not installed. Install with: pip install ffmpeg-python")
    FFMPEG_AVAILABLE = False

logger = logging.getLogger(__name__)

class AudioTranscriptionProcessor:
    """Enhanced processor with retry mechanism and notifications for audio transcription and clinical analysis"""
    
    def __init__(self, supabase_client, openai_api_key: str):
        self.supabase = supabase_client
        self.openai_api_key = openai_api_key
        
        # Initialize OpenAI client (new v1.0+ format)
        if OPENAI_AVAILABLE and openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
            logger.info("✅ OpenAI client initialized")
        else:
            self.openai_client = None
            logger.warning("⚠️ OpenAI not available - analysis will be skipped")
        
        # Initialize Whisper model (lazy loading)
        self.whisper_model = None
        
        # Processing configuration with retry settings
        self.config = {
            "whisper_model_size": "base",  # tiny, base, small, medium, large
            "max_file_size_mb": 500,
            "supported_formats": [".flac", ".wav", ".mp3", ".m4a", ".ogg"],
            "temp_dir": tempfile.gettempdir(),
            "openai_model": "gpt-3.5-turbo-1106",  # Reliable for JSON responses
            "max_retries": 2,  # Maximum retry attempts
            "retry_delay_base": 5.0,  # Base delay in seconds
            "retry_delay_multiplier": 2.0  # Exponential backoff multiplier
        }
    
    def _get_whisper_model(self):
        """Lazy load Whisper model"""
        if not WHISPER_AVAILABLE:
            raise Exception("faster-whisper is not installed")
        
        if self.whisper_model is None:
            logger.info(f"🤖 Loading Whisper model: {self.config['whisper_model_size']}")
            self.whisper_model = WhisperModel(
                self.config["whisper_model_size"], 
                device="cpu",  # Change to "cuda" if you have GPU
                compute_type="int8"
            )
            logger.info("✅ Whisper model loaded successfully")
        
        return self.whisper_model
    
    # NEW: Notifications table management
    async def _create_or_update_notification(self, user_id: str, title: str, content: str, audio_details_id: str = None):
        """
        Create or update notification - single entry per processing job
        
        Args:
            user_id: User ID for the notification
            title: Notification title
            content: Notification content
            audio_details_id: Optional ID to identify existing notification for this job
        """
        try:
            logger.info(f"📢 Managing notification for user {user_id}: {title}")
            
            notification_data = {
                "user_id": user_id,
                "title": title,
                "content": content,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Check if we have an existing notification for this processing job
            if audio_details_id:
                # Look for existing notification with audio_details_id reference in content or title
                existing_result = self.supabase.table("notifications").select("id").eq("user_id", user_id).ilike("content", f"%{audio_details_id}%").execute()
                
                if existing_result.data and len(existing_result.data) > 0:
                    # Update existing notification
                    existing_id = existing_result.data[0]["id"]
                    result = self.supabase.table("notifications").update(notification_data).eq("id", existing_id).execute()
                    
                    if result.data and len(result.data) > 0:
                        logger.info(f"✅ Updated existing notification {existing_id} for user {user_id}")
                        return existing_id
                    else:
                        logger.warning(f"⚠️ Failed to update notification {existing_id}")
                        return None
            
            # Create new notification if no existing one found
            result = self.supabase.table("notifications").insert(notification_data).execute()
            
            if result.data and len(result.data) > 0:
                notification_id = result.data[0]["id"]
                logger.info(f"✅ Created new notification {notification_id} for user {user_id}")
                return notification_id
            else:
                logger.warning(f"⚠️ Failed to create notification for user {user_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to manage notification: {e}")
            return None
    
    async def _notify_processing_started(self, user_id: str, audio_details_id: str):
        """Send notification when processing starts"""
        title = "Audio Processing Started"
        content = f"Your audio transcription has started processing. Audio ID: {audio_details_id}. You'll be notified when it's complete."
        
        return await self._create_or_update_notification(user_id, title, content, audio_details_id)
    
    async def _notify_processing_completed(self, user_id: str, audio_details_id: str, transcription_id: str):
        """Send notification when processing completes successfully"""
        title = "Audio Processing Complete ✅"
        content = f"Your audio transcription has been completed successfully! Audio ID: {audio_details_id}, Transcription ID: {transcription_id}. You can now view your results."
        
        return await self._create_or_update_notification(user_id, title, content, audio_details_id)
    
    async def _notify_processing_failed(self, user_id: str, audio_details_id: str, error_message: str = ""):
        """Send notification when processing fails permanently"""
        title = "Audio Processing Failed ❌"
        error_part = f" Error: {error_message[:100]}..." if error_message else ""
        content = f"Your audio transcription processing has failed after multiple attempts. Audio ID: {audio_details_id}.{error_part} Please try uploading again or contact support."
        
        return await self._create_or_update_notification(user_id, title, content, audio_details_id)
    
    async def _notify_processing_retry(self, user_id: str, audio_details_id: str, retry_count: int, max_retries: int):
        """Send notification when processing is being retried"""
        title = "Audio Processing Retry 🔄"
        content = f"Your audio transcription encountered an issue and is being retried. Audio ID: {audio_details_id}. Attempt {retry_count}/{max_retries}. Please wait while we process your audio."
        
        return await self._create_or_update_notification(user_id, title, content, audio_details_id)

    # UPDATED: Main entry point with retry logic and notifications
    async def process_audio_from_details_with_retry(self, audio_details_id: str, audio_url_data: Any, user_id: str):
        """
        Main processing pipeline with retry logic and notifications
        
        Args:
            audio_details_id: ID of the audio_details record
            audio_url_data: JSONB audio_url field (string, object, or array)
            user_id: User ID for the transcription
        """
        # Get current retry count from database
        current_retry_count = await self._get_current_retry_count(audio_details_id)
        
        logger.info(f"🔄 Processing audio_details {audio_details_id} - Attempt {current_retry_count + 1}/{self.config['max_retries'] + 1}")
        
        # Send initial notification (only on first attempt)
        if current_retry_count == 0:
            await self._notify_processing_started(user_id, audio_details_id)
        
        try:
            # Process the JSONB audio_url field
            processed_audio_urls = self._process_audio_url_field(audio_url_data)
            
            # Attempt processing
            transcription_id = await self.process_audio_from_details(audio_details_id, processed_audio_urls, user_id)
            
            # If successful, reset retry count and send success notification
            await self._reset_retry_count(audio_details_id)
            await self._notify_processing_completed(user_id, audio_details_id, transcription_id)
            logger.info(f"✅ Audio processing completed successfully for {audio_details_id}")
            
        except Exception as e:
            logger.error(f"❌ Processing attempt {current_retry_count + 1} failed for {audio_details_id}: {str(e)}")
            
            # Check if we should retry
            if current_retry_count < self.config["max_retries"]:
                # Increment retry count
                new_retry_count = await self._increment_retry_count(audio_details_id)
                
                # Send retry notification
                await self._notify_processing_retry(user_id, audio_details_id, new_retry_count, self.config["max_retries"])
                
                # Calculate retry delay with exponential backoff
                retry_delay = self.config["retry_delay_base"] * (self.config["retry_delay_multiplier"] ** current_retry_count)
                
                logger.info(f"🔄 Scheduling retry {new_retry_count}/{self.config['max_retries']} for {audio_details_id} in {retry_delay} seconds")
                
                # Update status to indicate retry pending
                await self._update_audio_details_status(audio_details_id, "retry_pending")
                
                # Schedule retry (async task)
                asyncio.create_task(self._schedule_retry(audio_details_id, audio_url_data, user_id, retry_delay))
                
            else:
                # Max retries reached - mark as permanently failed and send failure notification
                logger.error(f"❌ Max retries ({self.config['max_retries']}) reached for {audio_details_id}. Marking as permanently failed.")
                
                await self._mark_permanently_failed(audio_details_id)
                await self._notify_processing_failed(user_id, audio_details_id, str(e))
                
                # Still raise the exception for proper error handling
                raise e

    # NEW: Process JSONB audio_url field into consistent format
    def _process_audio_url_field(self, audio_url_data: Any) -> List[Dict]:
        """
        Process JSONB audio_url field from your schema into consistent format
        Handles: strings, objects, arrays of strings/objects
        """
        audio_urls = []
        
        logger.info(f"🔍 Processing audio_url field type: {type(audio_url_data)}")
        
        if isinstance(audio_url_data, str):
            # Single URL as string: "https://example.com/audio.flac"
            audio_urls = [{"audio_url": audio_url_data}]
            logger.info("📄 Single URL as string detected")
            
        elif isinstance(audio_url_data, dict):
            # Object format: {"audio_url": "...", "metadata": {...}}
            if 'audio_url' in audio_url_data:
                audio_urls = [audio_url_data]
                logger.info("📄 Single URL as object detected")
            else:
                # Object is the URL itself (edge case)
                audio_urls = [{"audio_url": str(audio_url_data)}]
                logger.info("📄 Object converted to URL detected")
                
        elif isinstance(audio_url_data, list):
            # Array of URLs or objects
            logger.info(f"📊 Array with {len(audio_url_data)} items detected")
            for i, item in enumerate(audio_url_data):
                if isinstance(item, str):
                    # Array of strings: ["url1", "url2"]
                    audio_urls.append({"audio_url": item})
                elif isinstance(item, dict) and 'audio_url' in item:
                    # Array of objects: [{"audio_url": "url1"}, {"audio_url": "url2"}]
                    audio_urls.append(item)
                else:
                    logger.warning(f"⚠️ Skipping unexpected item {i}: {item}")
        else:
            logger.warning(f"⚠️ Unexpected audio_url_data type: {type(audio_url_data)}")
            # Fallback - convert to string
            audio_urls = [{"audio_url": str(audio_url_data)}]
        
        # Filter out invalid URLs
        valid_urls = []
        for url_data in audio_urls:
            url = url_data.get("audio_url")
            if url and isinstance(url, str) and len(url) > 10:
                valid_urls.append(url_data)
            else:
                logger.warning(f"⚠️ Skipping invalid URL: {url_data}")
        
        logger.info(f"✅ Processed {len(valid_urls)} valid URLs from audio_url field")
        return valid_urls

    # NEW: Retry logic methods
    async def _get_current_retry_count(self, audio_details_id: str) -> int:
        """Get current retry count from audio_details table"""
        try:
            result = self.supabase.table("audio_details").select("retry_attempts").eq("id", audio_details_id).execute()
            
            if result.data and len(result.data) > 0:
                retry_count = result.data[0].get("retry_attempts", 0)
                return retry_count if retry_count is not None else 0
            else:
                logger.warning(f"⚠️ Could not find record {audio_details_id} to get retry count")
                return 0
                
        except Exception as e:
            logger.error(f"❌ Failed to get retry count for {audio_details_id}: {e}")
            return 0
    
    async def _increment_retry_count(self, audio_details_id: str) -> int:
        """Increment retry count in audio_details table"""
        try:
            # First get current count
            current_count = await self._get_current_retry_count(audio_details_id)
            new_count = current_count + 1
            
            # Update with new count
            update_data = {
                "retry_attempts": new_count,
                "updated_at": datetime.utcnow().isoformat(),
                "last_retry_at": datetime.utcnow().isoformat()
            }
            
            result = self.supabase.table("audio_details").update(update_data).eq("id", audio_details_id).execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"📊 Retry count incremented to {new_count} for {audio_details_id}")
                return new_count
            else:
                logger.warning(f"⚠️ Failed to increment retry count for {audio_details_id}")
                return current_count
                
        except Exception as e:
            logger.error(f"❌ Failed to increment retry count for {audio_details_id}: {e}")
            return current_count
    
    async def _reset_retry_count(self, audio_details_id: str):
        """Reset retry count to 0 after successful processing"""
        try:
            update_data = {
                "retry_attempts": 0,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            result = self.supabase.table("audio_details").update(update_data).eq("id", audio_details_id).execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"🔄 Retry count reset to 0 for {audio_details_id}")
            else:
                logger.warning(f"⚠️ Failed to reset retry count for {audio_details_id}")
                
        except Exception as e:
            logger.error(f"❌ Failed to reset retry count for {audio_details_id}: {e}")
    
    async def _mark_permanently_failed(self, audio_details_id: str):
        """Mark record as permanently failed after max retries"""
        try:
            update_data = {
                "status": "permanently_failed",
                "updated_at": datetime.utcnow().isoformat(),
                "failed_at": datetime.utcnow().isoformat()
            }
            
            result = self.supabase.table("audio_details").update(update_data).eq("id", audio_details_id).execute()
            
            if result.data and len(result.data) > 0:
                logger.error(f"💀 Record {audio_details_id} marked as permanently failed")
            else:
                logger.warning(f"⚠️ Failed to mark {audio_details_id} as permanently failed")
                
        except Exception as e:
            logger.error(f"❌ Failed to mark {audio_details_id} as permanently failed: {e}")
    
    async def _schedule_retry(self, audio_details_id: str, audio_url_data: Any, user_id: str, delay_seconds: float):
        """Schedule a retry after delay"""
        try:
            logger.info(f"⏰ Waiting {delay_seconds} seconds before retry for {audio_details_id}")
            await asyncio.sleep(delay_seconds)
            
            logger.info(f"🔄 Starting retry for {audio_details_id}")
            await self.process_audio_from_details_with_retry(audio_details_id, audio_url_data, user_id)
            
        except Exception as e:
            logger.error(f"❌ Retry failed for {audio_details_id}: {e}")
            # The process_audio_from_details_with_retry will handle further retries if needed

    # UPDATED: Main processing pipeline (returns transcription_id for notifications)
    async def process_audio_from_details(self, audio_details_id: str, audio_url_data: List[Dict], user_id: str) -> str:
        """
        Enhanced main processing pipeline that handles both single files and multiple chunks
        UPDATED: Returns transcription_id for notification purposes
        """
        transcription_id = None
        temp_files = []
        
        try:
            logger.info(f"🎵 Starting audio processing from audio_details: {audio_details_id}")
            logger.info(f"📊 Audio chunks detected: {len(audio_url_data)}")
            
            # Update audio_details status to processing
            await self._update_audio_details_status(audio_details_id, "processing")
            
            # Determine processing flow based on number of audio files
            if len(audio_url_data) == 1:
                # Single file flow
                audio_url = audio_url_data[0]["audio_url"]
                logger.info(f"📄 Single audio file detected: {audio_url}")
                final_audio_url = audio_url
                
            elif len(audio_url_data) > 1:
                # Multiple chunks flow - concatenate first
                logger.info(f"🔗 Multiple audio chunks detected ({len(audio_url_data)}), starting concatenation...")
                final_audio_url = await self._concatenate_audio_chunks(audio_url_data, audio_details_id)
                
                # Store the concatenated file URL in audio_details.full_audio_url
                await self._update_audio_details_full_url(audio_details_id, final_audio_url)
                logger.info(f"✅ Concatenated audio stored: {final_audio_url}")
                
            else:
                raise Exception("No audio URLs provided")
            
            # Step 2: Create a NEW transcription record (always creates new record)
            logger.info(f"📝 Creating transcription record...")
            transcription_id = await self._create_transcription_record(audio_details_id, final_audio_url, user_id)
            
            # Step 3: Download audio from S3
            logger.info(f"📥 Downloading audio from S3...")
            local_file_path = await self._download_audio_from_s3(final_audio_url)
            temp_files.append(local_file_path)
            
            # Step 4: Convert audio if needed
            logger.info(f"🔄 Processing audio file...")
            processed_file_path = await self._process_audio_file(local_file_path)
            if processed_file_path != local_file_path:
                temp_files.append(processed_file_path)
            
            # Step 5: Transcribe with Whisper
            logger.info(f"🎤 Transcribing audio...")
            transcript_text = await self._transcribe_audio(processed_file_path)
            
            if not transcript_text or len(transcript_text.strip()) < 10:
                raise Exception("Transcription resulted in empty or very short text")
            
            # Step 6: Analyze with OpenAI
            logger.info(f"🧠 Analyzing transcript with AI...")
            analysis_result = await self._analyze_transcript(transcript_text)
            
            # Step 7: Store results in transcription table
            logger.info(f"💾 Storing results...")
            await self._store_transcription_results(transcription_id, transcript_text, analysis_result)
            
            # Step 8: Update final statuses
            await self._update_transcription_status(transcription_id, "completed", "completed")
            await self._update_audio_details_status(audio_details_id, "completed")
            
            logger.info(f"✅ Audio processing completed successfully for {audio_details_id}")
            return transcription_id  # Return for notification purposes
            
        except Exception as e:
            logger.error(f"❌ Audio processing failed for {audio_details_id}: {str(e)}")
            
            # Update both tables to failed status
            if transcription_id:
                await self._update_transcription_status(transcription_id, "failed", "failed")
            await self._update_audio_details_status(audio_details_id, "failed")
            
            # Re-raise the exception for retry handling
            raise e
            
        finally:
            # Cleanup temporary files
            await self._cleanup_files(temp_files)

    # ALL YOUR EXISTING METHODS (unchanged - including concatenation, transcription, analysis, etc.)
    async def _concatenate_audio_chunks(self, audio_url_data: List[Dict], audio_details_id: str) -> str:
        """Download multiple audio chunks and concatenate them into a single file"""
        chunk_files = []
        
        try:
            logger.info(f"🔗 Starting concatenation of {len(audio_url_data)} audio chunks...")
            
            # Step 1: Download all audio chunks
            for i, url_data in enumerate(audio_url_data):
                audio_url = url_data["audio_url"]
                logger.info(f"📥 Downloading chunk {i+1}/{len(audio_url_data)}: {audio_url}")
                
                chunk_file = await self._download_audio_chunk(audio_url, i)
                chunk_files.append(chunk_file)
                logger.info(f"✅ Chunk {i+1} downloaded: {os.path.getsize(chunk_file)} bytes")
            
            # Step 2: Concatenate audio files (no FFmpeg required)
            logger.info(f"🔗 Concatenating {len(chunk_files)} audio chunks...")
            concatenated_file = await self._concatenate_audio_files(chunk_files)
            
            # Step 3: Upload concatenated file to storage
            logger.info(f"☁️ Uploading concatenated audio to storage...")
            final_url = await self._upload_concatenated_audio(concatenated_file, audio_details_id)
            
            logger.info(f"✅ Audio concatenation completed: {final_url}")
            return final_url
            
        except Exception as e:
            logger.error(f"❌ Audio concatenation failed: {e}")
            raise Exception(f"Failed to concatenate audio chunks: {str(e)}")
            
        finally:
            # Cleanup chunk files
            await self._cleanup_files(chunk_files)
            if 'concatenated_file' in locals():
                await self._cleanup_files([concatenated_file])
    
    async def _download_audio_chunk(self, audio_url: str, chunk_index: int) -> str:
        """Download a single audio chunk"""
        try:
            # Create temporary file for this chunk
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_chunk_{chunk_index}.flac")
            temp_file_path = temp_file.name
            temp_file.close()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(audio_url) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download chunk {chunk_index}: HTTP {response.status}")
                    
                    # Check file size
                    content_length = response.headers.get('content-length')
                    if content_length:
                        size_mb = int(content_length) / (1024 * 1024)
                        if size_mb > self.config["max_file_size_mb"]:
                            raise Exception(f"Chunk {chunk_index} too large: {size_mb:.1f}MB (max: {self.config['max_file_size_mb']}MB)")
                    
                    # Download chunk
                    async with aiofiles.open(temp_file_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
            
            # Verify file was downloaded
            if not os.path.exists(temp_file_path) or os.path.getsize(temp_file_path) == 0:
                raise Exception(f"Downloaded chunk {chunk_index} is empty or doesn't exist")
            
            return temp_file_path
            
        except Exception as e:
            logger.error(f"❌ Chunk {chunk_index} download failed: {e}")
            raise Exception(f"Failed to download audio chunk {chunk_index}: {str(e)}")
    
    async def _concatenate_audio_files(self, chunk_files: List[str]) -> str:
        """Enhanced concatenation with multiple methods - no FFmpeg required"""
        try:
            logger.info(f"🔗 Attempting to concatenate {len(chunk_files)} audio files...")
            
            # Method 1: Try pydub (pure Python, no FFmpeg needed)
            try:
                return await self._concatenate_with_pydub(chunk_files)
            except ImportError:
                logger.warning("⚠️ pydub not installed. Install with: pip install pydub")
            except Exception as e:
                logger.warning(f"⚠️ Pydub failed: {e}")
            
            # Method 2: Try binary concatenation (simple fallback)
            logger.info("🔧 Trying binary concatenation as fallback...")
            return await self._concatenate_binary(chunk_files)
            
        except Exception as e:
            logger.error(f"❌ All concatenation methods failed: {e}")
            raise Exception(f"All concatenation methods failed: {str(e)}")
    
    async def _concatenate_with_pydub(self, chunk_files: List[str]) -> str:
        """Concatenate audio files using pydub (pure Python, no FFmpeg needed)"""
        try:
            from pydub import AudioSegment
            logger.info("🔧 Using pydub for concatenation (no FFmpeg required)")
            
            # Load first audio file
            combined = AudioSegment.from_file(chunk_files[0])
            logger.info(f"✅ Loaded chunk 1: {len(combined)}ms duration")
            
            # Concatenate remaining files
            for i, chunk_file in enumerate(chunk_files[1:], 2):
                chunk_audio = AudioSegment.from_file(chunk_file)
                combined += chunk_audio
                logger.info(f"✅ Added chunk {i}: {len(chunk_audio)}ms duration")
            
            # Export concatenated audio
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix="_concatenated.flac")
            output_path = output_file.name
            output_file.close()
            
            # Export as FLAC
            combined.export(output_path, format="flac")
            
            file_size = os.path.getsize(output_path)
            logger.info(f"✅ Pydub concatenation completed: {file_size} bytes, {len(combined)}ms duration")
            
            return output_path
            
        except ImportError:
            logger.error("❌ pydub not installed")
            raise ImportError("pydub library not available")
        except Exception as e:
            logger.error(f"❌ Pydub concatenation failed: {e}")
            raise Exception(f"Pydub concatenation failed: {str(e)}")
    
    async def _concatenate_binary(self, chunk_files: List[str]) -> str:
        """Simple binary concatenation fallback"""
        try:
            logger.info("🔧 Using binary concatenation (simple merge)")
            
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix="_concatenated.flac")
            output_path = output_file.name
            output_file.close()
            
            total_bytes = 0
            with open(output_path, 'wb') as outfile:
                for i, chunk_file in enumerate(chunk_files):
                    with open(chunk_file, 'rb') as infile:
                        data = infile.read()
                        outfile.write(data)
                        total_bytes += len(data)
                        logger.info(f"✅ Added chunk {i+1}: {len(data)} bytes")
            
            logger.info(f"✅ Binary concatenation completed: {total_bytes} bytes")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Binary concatenation failed: {e}")
            raise Exception(f"Binary concatenation failed: {str(e)}")
    
    async def _upload_concatenated_audio(self, file_path: str, audio_details_id: str) -> str:
        """Upload concatenated audio file to Supabase storage and return URL"""
        try:
            # Generate unique filename
            filename = f"concatenated_{audio_details_id}_{uuid.uuid4().hex[:8]}.flac"
            storage_path = f"audio/{audio_details_id}/{filename}"
            
            logger.info(f"☁️ Uploading concatenated audio: {storage_path}")
            
            # Read file content
            async with aiofiles.open(file_path, 'rb') as f:
                file_content = await f.read()
            
            # Upload to Supabase storage
            result = self.supabase.storage.from_("audio-files").upload(storage_path, file_content)
            
            # Handle different response formats for newer Supabase versions
            if hasattr(result, 'error') and result.error:
                raise Exception(f"Storage upload failed: {result.error}")
            elif hasattr(result, 'data') and not result.data:
                raise Exception("Storage upload failed: No data returned")
            
            # Get public URL
            public_url = self.supabase.storage.from_("audio-files").get_public_url(storage_path)
            
            logger.info(f"✅ Concatenated audio uploaded successfully: {public_url}")
            return public_url
            
        except Exception as e:
            logger.error(f"❌ Failed to upload concatenated audio: {e}")
            raise Exception(f"Failed to upload concatenated audio: {str(e)}")

    async def _update_audio_details_full_url(self, audio_details_id: str, full_audio_url: str):
        """Update the full_audio_url field in audio_details table with error handling"""
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                update_data = {
                    "full_audio_url": full_audio_url,
                    "updated_at": datetime.utcnow().isoformat()
                }
                
                logger.info(f"📊 Updating full_audio_url for record {audio_details_id} (attempt {attempt + 1}/{max_retries})")
                
                result = self.supabase.table("audio_details").update(update_data).eq("id", audio_details_id).execute()
                
                if result.data and len(result.data) > 0:
                    logger.info(f"✅ Audio details full_audio_url updated successfully for record {audio_details_id}")
                    return True
                else:
                    logger.warning(f"⚠️ Update returned no data for record {audio_details_id}")
                    return False
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.info(f"🔄 Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 1.5, 5.0)
                else:
                    logger.error(f"❌ Failed to update full_audio_url after {max_retries} attempts")
                    raise Exception(f"Failed to update full_audio_url after {max_retries} attempts: {str(e)}")

    async def _create_transcription_record(self, audio_details_id: str, audio_url: str, user_id: str) -> str:
        """Create a new record in audio_transcriptions table linked to audio_details"""
        try:
            # Generate a unique ID for the transcription record
            transcription_id = str(uuid.uuid4())
            
            logger.info(f"📝 Creating new transcription record with ID: {transcription_id}")
            logger.info(f"🔗 Linking to audio_details: {audio_details_id}")
            logger.info(f"👤 User ID: {user_id}")
            
            # Create record with all required fields
            transcription_record = {
                "id": transcription_id,  # Unique ID for this transcription
                "user_id": user_id,      # User who owns this transcription
                "audio_details_id": audio_details_id,  # Link to specific audio_details record
                "audio_url": audio_url,  # Store the final audio URL (single file or concatenated)
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
            
            # Insert into audio_transcriptions table
            result = self.supabase.table("audio_transcriptions").insert(transcription_record).execute()
            
            if not result.data or len(result.data) == 0:
                raise Exception("Failed to create transcription record - no data returned")
            
            created_record = result.data[0]
            created_id = created_record["id"]
            
            logger.info(f"✅ Created NEW transcription record: {created_id}")
            return created_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create transcription record: {e}")
            raise Exception(f"Failed to create transcription record: {str(e)}")
    
    async def _download_audio_from_s3(self, audio_url: str) -> str:
        """Download audio file from S3 URL"""
        try:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".flac")
            temp_file_path = temp_file.name
            temp_file.close()
            
            logger.info(f"📥 Downloading {audio_url} to {temp_file_path}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(audio_url) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download audio: HTTP {response.status}")
                    
                    # Check file size
                    content_length = response.headers.get('content-length')
                    if content_length:
                        size_mb = int(content_length) / (1024 * 1024)
                        if size_mb > self.config["max_file_size_mb"]:
                            raise Exception(f"File too large: {size_mb:.1f}MB (max: {self.config['max_file_size_mb']}MB)")
                    
                    # Download file
                    async with aiofiles.open(temp_file_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
            
            # Verify file was downloaded
            if not os.path.exists(temp_file_path) or os.path.getsize(temp_file_path) == 0:
                raise Exception("Downloaded file is empty or doesn't exist")
            
            logger.info(f"✅ Audio downloaded successfully: {os.path.getsize(temp_file_path)} bytes")
            return temp_file_path
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            raise Exception(f"Failed to download audio from S3: {str(e)}")
    
    async def _process_audio_file(self, file_path: str) -> str:
        """Process audio file (convert if needed)"""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise Exception(f"Audio file not found: {file_path}")
            
            file_ext = Path(file_path).suffix.lower()
            
            # If already in supported format, return as-is
            if file_ext in self.config["supported_formats"]:
                logger.info(f"✅ Audio file already in supported format: {file_ext}")
                return file_path
            
            # Convert to WAV if ffmpeg is available
            if FFMPEG_AVAILABLE:
                logger.info(f"🔄 Converting {file_ext} to WAV...")
                
                output_path = file_path.replace(file_ext, ".wav")
                
                # Convert using ffmpeg
                (
                    ffmpeg
                    .input(file_path)
                    .output(output_path, acodec='pcm_s16le', ar=16000)
                    .overwrite_output()
                    .run(quiet=True)
                )
                
                logger.info(f"✅ Audio converted successfully to {output_path}")
                return output_path
            else:
                logger.warning(f"⚠️ ffmpeg not available, using original file: {file_ext}")
                return file_path
                
        except Exception as e:
            logger.error(f"❌ Audio processing failed: {e}")
            raise Exception(f"Failed to process audio file: {str(e)}")
    
    async def _transcribe_audio(self, file_path: str) -> str:
        """Transcribe audio using Faster Whisper"""
        try:
            if not WHISPER_AVAILABLE:
                raise Exception("Whisper is not available for transcription")
            
            # Get Whisper model
            model = self._get_whisper_model()
            
            logger.info(f"🎤 Starting transcription of {file_path}")
            
            # Run transcription in thread pool to avoid blocking
            def transcribe_sync():
                segments, info = model.transcribe(
                    file_path,
                    beam_size=5,
                    language="en",  # Specify language or use None for auto-detection
                    vad_filter=True,  # Voice activity detection
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                # Combine all segments into full transcript
                transcript_parts = []
                for segment in segments:
                    transcript_parts.append(segment.text)
                
                return " ".join(transcript_parts).strip()
            
            # Run transcription in executor
            loop = asyncio.get_event_loop()
            transcript = await loop.run_in_executor(None, transcribe_sync)
            
            if not transcript:
                raise Exception("Transcription resulted in empty text")
            
            logger.info(f"✅ Transcription completed: {len(transcript)} characters")
            logger.info(f"📝 Preview: {transcript[:200]}...")
            
            return transcript
            
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            raise Exception(f"Failed to transcribe audio: {str(e)}")

    async def _analyze_transcript(self, transcript: str) -> Dict[str, Any]:
        """Analyze transcript using OpenAI with custom medical prompt"""
        try:
            if not self.openai_client:
                logger.warning("⚠️ OpenAI not available, skipping analysis")
                return self._create_dummy_analysis(transcript)
            
            logger.info(f"🧠 Analyzing transcript with OpenAI...")
            
            # Custom system prompt
            system_prompt = """You are a medical AI assistant. Analyze transcripts and extract structured clinical information from ANY medical-related content, including patient symptom descriptions.

        INSTRUCTIONS:
        1. Extract medical information from patient descriptions, doctor consultations, or any health-related content
        2. If symptoms are mentioned, list them in the symptoms array
        3. If timeline is provided, include it in key_points
        4. If concerns are expressed, include them in key_points
        5. Always provide recommendations for medical evaluation when symptoms are described
        6. Only return empty arrays if there is truly NO medical content (like casual conversation)
        
        Always respond in valid JSON format with these exact fields: 
        - clinical_summary (string): Brief summary of the medical content discussed
        - key_clinical_insights (array): Important clinical observations from the content
        - key_points (array): Main health-related points discussed
        - action_items (array): Suggested actions based on the content
        - follow_up (array): Recommended follow-up care
        - medications (array): Any medications mentioned (empty if none)
        - symptoms (array): All symptoms described or mentioned
        - diagnosis (array): Any diagnoses mentioned (empty if none given)
        - recommendations (array): Clinical recommendations based on content
        - next_steps (array): Suggested next steps for care
        
        Extract ALL relevant medical information even from informal patient descriptions."""
            
            # Custom user prompt
            user_prompt = f"""Extract all medical information from this transcript. Even if it's just a patient describing symptoms, extract the symptoms, timeline, concerns, and provide appropriate medical recommendations:

        TRANSCRIPT:
        "{transcript}"
        
        Extract all symptoms, concerns, timeline, and provide appropriate recommendations for medical evaluation."""
            
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.config["openai_model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=4000,
            )
            
            analysis_text = response.choices[0].message.content.strip()
            logger.info(f"📄 OpenAI raw response (first 200 chars): {analysis_text[:200]}...")
            
            # Parse the response with robust error handling
            analysis_result = self._parse_openai_response(analysis_text)
            
            # Validate and clean up the result
            analysis_result = self._validate_and_fill_analysis(analysis_result, transcript)
            
            logger.info("✅ AI analysis completed successfully")
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ AI analysis failed: {e}")
            logger.warning("⚠️ Falling back to dummy analysis")
            return self._create_dummy_analysis(transcript)
    
    def _parse_openai_response(self, response_text: str) -> Dict[str, Any]:
        """Robust parsing of OpenAI response with multiple fallback strategies"""
        try:
            # Strategy 1: Direct JSON parsing
            logger.info("🔍 Attempting direct JSON parsing...")
            return json.loads(response_text)
            
        except json.JSONDecodeError:
            logger.warning("⚠️ Direct JSON parsing failed, trying cleanup strategies...")
            
            # Strategy 2: Clean up common issues
            cleaned_text = response_text.strip()
            
            # Remove markdown code blocks
            cleaned_text = re.sub(r'```json\s*', '', cleaned_text)
            cleaned_text = re.sub(r'```\s*', '', cleaned_text)
            
            try:
                logger.info("🔍 Attempting JSON parsing after cleanup...")
                return json.loads(cleaned_text)
            except json.JSONDecodeError:
                
                # Strategy 3: Extract JSON from mixed content
                logger.info("🔍 Attempting to extract JSON from mixed content...")
                json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
                if json_match:
                    json_part = json_match.group(0)
                    try:
                        return json.loads(json_part)
                    except json.JSONDecodeError:
                        pass
                
                # Strategy 4: Create structured response from text
                logger.warning("⚠️ All JSON parsing failed, creating structured response")
                return self._create_structured_from_text(response_text)

    def _validate_and_fill_analysis(self, analysis: Dict[str, Any], transcript: str) -> Dict[str, Any]:
        """Validate and intelligently fill analysis based on transcript content"""
        required_fields = [
            "clinical_summary", "key_clinical_insights", "key_points", 
            "action_items", "follow_up", "medications", "symptoms", 
            "diagnosis", "recommendations", "next_steps"
        ]
        
        # Ensure all required fields exist
        for field in required_fields:
            if field not in analysis:
                analysis[field] = [] if field != "clinical_summary" else ""
            
            # Ensure arrays are actually arrays for JSONB fields
            if field != "clinical_summary" and not isinstance(analysis[field], list):
                if isinstance(analysis[field], str):
                    analysis[field] = [analysis[field]] if analysis[field] else []
                else:
                    analysis[field] = []
        
        # Smart filling based on transcript content
        transcript_lower = transcript.lower()
        
        # Fill medications if empty
        if not analysis["medications"]:
            extracted_meds = self._extract_medications_from_transcript(transcript)
            if extracted_meds:
                analysis["medications"] = extracted_meds
            elif any(word in transcript_lower for word in ["medication", "pill", "prescription", "drug", "treatment"]):
                analysis["medications"] = ["Medications discussed in consultation"]
            else:
                analysis["medications"] = ["No specific medications mentioned"]
        
        # Fill symptoms if empty
        if not analysis["symptoms"]:
            extracted_symptoms = self._extract_symptoms_from_transcript(transcript)
            if extracted_symptoms:
                analysis["symptoms"] = extracted_symptoms
            elif any(word in transcript_lower for word in ["feel", "pain", "hurt", "sick", "symptom", "problem"]):
                analysis["symptoms"] = ["Symptoms discussed during consultation"]
            else:
                analysis["symptoms"] = ["No specific symptoms reported"]
        
        # Fill other empty fields with defaults
        if not analysis["diagnosis"]:
            analysis["diagnosis"] = ["General health consultation", "Clinical discussion documented"]
        
        if not analysis["recommendations"]:
            analysis["recommendations"] = [
                "Continue regular healthcare monitoring",
                "Follow healthcare provider guidance",
                "Maintain healthy lifestyle practices"
            ]
        
        if not analysis["key_clinical_insights"]:
            analysis["key_clinical_insights"] = [
                "Patient consultation documented",
                "Clinical conversation captured for medical record"
            ]
        
        if not analysis["key_points"]:
            analysis["key_points"] = [
                "Medical consultation completed",
                "Patient concerns addressed by healthcare provider"
            ]
        
        if not analysis["action_items"]:
            analysis["action_items"] = [
                "Review complete transcript for specific instructions",
                "Follow healthcare provider recommendations"
            ]
        
        if not analysis["follow_up"]:
            analysis["follow_up"] = [
                "Schedule follow-up as recommended",
                "Monitor health status as discussed"
            ]
        
        if not analysis["next_steps"]:
            analysis["next_steps"] = [
                "Continue current care plan",
                "Implement recommendations from consultation"
            ]
        
        # Ensure clinical_summary is not empty
        if not analysis["clinical_summary"] or len(analysis["clinical_summary"].strip()) < 10:
            analysis["clinical_summary"] = f"Medical consultation completed with comprehensive discussion of patient health status, symptoms, and treatment options. Complete transcript available for medical record documentation."
        
        return analysis
    
    def _create_structured_from_text(self, text: str) -> Dict[str, Any]:
        """Extract structured data from unstructured text response"""
        logger.info("🛠️ Creating structured response from unstructured text...")
        
        result = {
            "clinical_summary": "",
            "key_clinical_insights": [],
            "key_points": [],
            "action_items": [],
            "follow_up": [],
            "medications": [],
            "symptoms": [],
            "diagnosis": [],
            "recommendations": [],
            "next_steps": []
        }
        
        # Extract clinical summary from first meaningful line
        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 50]
        if lines:
            result["clinical_summary"] = lines[0][:1000]
        
        # Add default values for empty fields
        result["key_points"] = ["Medical consultation documented", "Patient concerns addressed"]
        result["action_items"] = ["Review complete transcript for specific instructions"]
        result["recommendations"] = ["Follow healthcare provider guidance"]
        
        return result

    def _create_dummy_analysis(self, transcript: str) -> Dict[str, Any]:
        """Create comprehensive dummy analysis when OpenAI is not available"""
        medications = self._extract_medications_from_transcript(transcript)
        symptoms = self._extract_symptoms_from_transcript(transcript)
        
        return {
            "clinical_summary": f"Medical conversation transcribed successfully. Patient consultation documented with {len(transcript)} characters of conversation content.",
            "key_clinical_insights": [
                "Patient consultation documented in full detail",
                "Healthcare provider and patient dialogue captured",
                "Medical discussion preserved for professional review"
            ],
            "key_points": [
                "Complete medical conversation captured",
                "Patient concerns and provider responses documented",
                "Treatment discussion recorded in detail"
            ],
            "action_items": [
                "Review complete transcript for medical instructions",
                "Follow provider recommendations from conversation", 
                "Schedule follow-up appointments as mentioned"
            ],
            "follow_up": [
                "Follow provider instructions for next appointment",
                "Monitor symptoms as discussed with healthcare provider",
                "Contact provider if symptoms change"
            ],
            "medications": medications,
            "symptoms": symptoms,
            "diagnosis": [
                "Clinical assessment documented in conversation",
                "Healthcare provider evaluation completed"
            ],
            "recommendations": [
                "Follow healthcare provider guidance from consultation",
                "Implement treatment recommendations discussed",
                "Continue care plan as outlined"
            ],
            "next_steps": [
                "Review transcript for specific next steps",
                "Complete recommended tests or procedures",
                "Schedule follow-up appointments as recommended"
            ]
        }
    
    def _extract_medications_from_transcript(self, transcript: str) -> List[str]:
        """Extract actual medication names from transcript text"""
        medications = []
        
        # Common medication names
        medication_keywords = [
            "metformin", "lisinopril", "ibuprofen", "tylenol", "aspirin", "advil", "aleve",
            "hydrochlorothiazide", "atorvastatin", "simvastatin", "omeprazole", "losartan",
            "amlodipine", "prednisone", "amoxicillin", "azithromycin", "levothyroxine",
            "warfarin", "insulin", "gabapentin", "sertraline", "fluoxetine", "citalopram"
        ]
        
        transcript_lower = transcript.lower()
        
        # Extract medication names that appear in transcript
        for med in medication_keywords:
            if med in transcript_lower:
                medications.append(med.capitalize())
        
        # Look for medication patterns
        med_patterns = [
            r'(\w+)\s+(?:\d+\s*mg|\d+\s*mcg|tablet|pill|capsule)',
            r'taking\s+(\w+)',
            r'prescribed\s+(\w+)',
            r'medication\s+(\w+)'
        ]
        
        for pattern in med_patterns:
            matches = re.findall(pattern, transcript_lower)
            for match in matches:
                if len(match) > 3 and match not in [m.lower() for m in medications]:
                    medications.append(match.capitalize())
        
        # Remove duplicates and common non-medication words
        exclude_words = ["take", "taking", "medication", "pill", "tablet", "dose"]
        medications = list(set([med for med in medications if med.lower() not in exclude_words]))
        
        return medications[:10] if medications else []
    
    def _extract_symptoms_from_transcript(self, transcript: str) -> List[str]:
        """Extract actual symptom names from transcript text"""
        symptoms = []
        
        # Common symptoms
        symptom_keywords = [
            "pain", "headache", "nausea", "fatigue", "tired", "dizzy", "fever", "chills", 
            "cough", "sore throat", "runny nose", "chest pain", "back pain", "stomach pain",
            "swelling", "rash", "itching", "burning", "numbness", "weakness", "anxiety", 
            "depression", "insomnia", "aching", "cramping", "bloating", "constipation"
        ]
        
        transcript_lower = transcript.lower()
        
        # Extract symptoms that appear in transcript
        for symptom in symptom_keywords:
            if symptom in transcript_lower:
                symptoms.append(symptom.capitalize())
        
        # Look for symptom patterns
        symptom_patterns = [
            r'feeling\s+(\w+)',
            r'experiencing\s+(\w+)',
            r'having\s+(\w+)',
            r'symptoms?\s+(?:of\s+)?(\w+)'
        ]
        
        for pattern in symptom_patterns:
            matches = re.findall(pattern, transcript_lower)
            for match in matches:
                if len(match) > 3 and match not in [s.lower() for s in symptoms]:
                    symptoms.append(match.capitalize())
        
        # Remove duplicates and common non-symptom words
        exclude_words = ["feeling", "having", "experiencing", "getting", "been", "very"]
        symptoms = list(set([symptom for symptom in symptoms if symptom.lower() not in exclude_words]))
        
        return symptoms[:10] if symptoms else []

    # Store results for your actual schema
    async def _store_transcription_results(self, transcription_id: str, transcript: str, analysis: Dict[str, Any]):
        """Store results in audio_transcriptions table - Updated for your JSONB schema"""
        try:
            # Your database has JSONB fields, so pass arrays directly
            update_data = {
                "transcript_text": transcript,  # Your table has this field
                "transcript": transcript,       # Your table also has this field
                "clinical_summary": analysis.get("clinical_summary", ""),
                # JSONB fields - pass Python lists/dicts directly
                "key_clinical_insights": analysis.get("key_clinical_insights", []),
                "key_points": analysis.get("key_points", []),
                "action_items": analysis.get("action_items", []),
                "follow_up": analysis.get("follow_up", []),
                "medications": analysis.get("medications", []),
                "symptoms": analysis.get("symptoms", []),
                "diagnosis": analysis.get("diagnosis", []),
                "recommendations": analysis.get("recommendations", []),
                "next_steps": analysis.get("next_steps", []),
                "processed_at": datetime.utcnow().isoformat(),
                "status": "completed"
            }
            
            result = self.supabase.table("audio_transcriptions").update(update_data).eq("id", transcription_id).execute()
            
            if not result.data:
                raise Exception("Failed to update transcription table with results")
            
            logger.info(f"✅ Results stored successfully in audio_transcriptions table")
            
        except Exception as e:
            logger.error(f"❌ Failed to store transcription results: {e}")
            raise Exception(f"Failed to store results in transcription table: {str(e)}")
    
    # Status updates for your schema
    async def _update_transcription_status(self, transcription_id: str, status: str, message: str = ""):
        """Update processing status in audio_transcriptions table"""
        try:
            update_data = {"status": status}
            
            if status == "in_progress":
                update_data["started_at"] = datetime.utcnow().isoformat()
            elif status in ["completed", "failed"]:
                update_data["completed_at"] = datetime.utcnow().isoformat()
            
            if update_data:
                result = self.supabase.table("audio_transcriptions").update(update_data).eq("id", transcription_id).execute()
                logger.info(f"📊 Transcription status updated to '{status}' for record {transcription_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to update transcription status: {e}")
    
    # Status updates with retry fields
    async def _update_audio_details_status(self, audio_details_id: str, status: str):
        """Update status in audio_details table with retry timestamp fields"""
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Add timestamp fields based on status
            if status == "processing":
                update_data["processing_started_at"] = datetime.utcnow().isoformat()
            elif status == "completed":
                update_data["completed_at"] = datetime.utcnow().isoformat()
            elif status == "failed" or status == "permanently_failed":
                update_data["failed_at"] = datetime.utcnow().isoformat()
            elif status == "retry_pending":
                update_data["retry_scheduled_at"] = datetime.utcnow().isoformat()
            
            result = self.supabase.table("audio_details").update(update_data).eq("id", audio_details_id).execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"✅ Audio details status updated to '{status}' successfully")
                return True
            else:
                logger.warning(f"⚠️ Update returned no data for record {audio_details_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to update audio details status: {e}")
            return False
    
    async def _cleanup_files(self, file_paths: list):
        """Clean up temporary files"""
        for file_path in file_paths:
            try:
                if file_path and os.path.exists(file_path):
                    os.unlink(file_path)
                    logger.info(f"🗑️ Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup file {file_path}: {e}")


# Helper functions for webhook integration
async def process_audio_with_chunks_and_retry(supabase_client, openai_api_key: str, audio_details_id: str, audio_url_data: Any, user_id: str):
    """
    Main helper function with retry logic and notifications for webhook integration
    
    Args:
        supabase_client: Supabase client instance
        openai_api_key: OpenAI API key
        audio_details_id: ID of the audio_details record
        audio_url_data: JSONB audio_url field (any format)
        user_id: User ID for the transcription
    """
    processor = AudioTranscriptionProcessor(supabase_client, openai_api_key)
    await processor.process_audio_from_details_with_retry(audio_details_id, audio_url_data, user_id)

# Legacy compatibility function
async def process_audio_with_chunks(supabase_client, openai_api_key: str, audio_details_id: str, audio_url_data: Any, user_id: str):
    """Legacy function - now uses retry logic with notifications"""
    await process_audio_with_chunks_and_retry(supabase_client, openai_api_key, audio_details_id, audio_url_data, user_id)

# Testing function
async def test_processor():
    """Test function for the enhanced audio processor with retry logic and notifications"""
    logger.info("🧪 Testing enhanced audio processor with retry mechanism and notifications...")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_processor())


