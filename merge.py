import os
import re
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

def extract_number_from_filename(filename: str) -> int:
    """Extract number from filename for sorting"""
    # Look for numbers in the filename
    numbers = re.findall(r'\d+', filename)
    
    if numbers:
        # Return the first number found (or last, depending on your preference)
        return int(numbers[0])
    else:
        # If no numbers found, return 0 (will be sorted first)
        return 0

def get_sorted_wav_files(directory_path: str) -> List[str]:
    """Get all .wav files sorted by numerical order in filename"""
    directory = Path(directory_path)
    
    if not directory.exists():
        logger.error(f"Directory does not exist: {directory_path}")
        return []
    
    # Get all .wav files
    wav_files = []
    for file_path in directory.glob("*.wav"):
        wav_files.append(str(file_path))
    
    if not wav_files:
        logger.warning(f"No .wav files found in: {directory_path}")
        return []
    
    # Sort by extracted numbers
    wav_files.sort(key=lambda x: extract_number_from_filename(Path(x).name))
    
    logger.info(f"Found {len(wav_files)} .wav files to combine")
    return wav_files

def combine_audio_files(input_directory: str, output_file: str = "combined_audio.wav", 
                       silence_duration: float = 0.5, target_sample_rate: int = 22050) -> bool:
    """
    Combine multiple WAV files from a directory in numerical order
    
    Args:
        input_directory: Path to directory containing .wav files
        output_file: Path for the combined output file
        silence_duration: Seconds of silence between files (default: 0.5s)
        target_sample_rate: Sample rate for output file (default: 22050 Hz)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get sorted list of WAV files
        wav_files = get_sorted_wav_files(input_directory)
        
        if not wav_files:
            logger.error("No WAV files found to combine")
            return False
        
        logger.info(f"Combining {len(wav_files)} files...")
        
        # List to store all audio data
        combined_audio = []
        
        # Generate silence to add between files
        silence_samples = int(silence_duration * target_sample_rate)
        silence = np.zeros(silence_samples, dtype=np.float32)
        
        for i, wav_file in enumerate(wav_files):
            try:
                # Read audio file
                audio_data, sample_rate = sf.read(wav_file)
                
                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                
                # Resample if necessary
                if sample_rate != target_sample_rate:
                    try:
                        import librosa
                        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sample_rate)
                    except ImportError:
                        logger.warning(f"librosa not available for resampling {wav_file}")
                        # Simple resampling fallback (not ideal but works)
                        factor = target_sample_rate / sample_rate
                        new_length = int(len(audio_data) * factor)
                        audio_data = np.interp(
                            np.linspace(0, len(audio_data), new_length),
                            np.arange(len(audio_data)),
                            audio_data
                        )
                
                # Normalize audio to prevent clipping
                audio_data = audio_data.astype(np.float32)
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
                
                # Add to combined audio
                combined_audio.append(audio_data)
                
                # Add silence between files (except after the last file)
                if i < len(wav_files) - 1:
                    combined_audio.append(silence)
                
                logger.info(f"Added: {Path(wav_file).name} ({len(audio_data)} samples)")
                
            except Exception as e:
                logger.error(f"Error processing {wav_file}: {e}")
                continue
        
        if not combined_audio:
            logger.error("No audio data to combine")
            return False
        
        # Concatenate all audio
        final_audio = np.concatenate(combined_audio)
        
        # Save combined audio
        sf.write(output_file, final_audio, target_sample_rate)
        
        total_duration = len(final_audio) / target_sample_rate
        logger.info(f"Successfully combined {len(wav_files)} files")
        logger.info(f"Output: {output_file} ({total_duration:.2f} seconds)")
        
        return True
        
    except Exception as e:
        logger.error(f"Error combining audio files: {e}")
        return False

def combine_audio_with_metadata(input_directory: str, output_file: str = "combined_audio.wav") -> dict:
    """
    Enhanced version that returns metadata about the combination process
    """
    result = {
        "success": False,
        "output_file": output_file,
        "files_processed": 0,
        "total_duration": 0.0,
        "file_list": []
    }
    
    try:
        wav_files = get_sorted_wav_files(input_directory)
        
        if not wav_files:
            return result
        
        # Print file order for verification
        print("File combination order:")
        for i, file_path in enumerate(wav_files):
            filename = Path(file_path).name
            number = extract_number_from_filename(filename)
            print(f"  {i+1:2d}. {filename} (extracted number: {number})")
            result["file_list"].append({
                "filename": filename,
                "path": file_path,
                "order": i+1,
                "extracted_number": number
            })
        
        # Combine the files
        success = combine_audio_files(input_directory, output_file)
        
        if success:
            # Get final file info
            if os.path.exists(output_file):
                audio_data, sample_rate = sf.read(output_file)
                result["total_duration"] = len(audio_data) / sample_rate
                result["files_processed"] = len(wav_files)
                result["success"] = True
        
        return result
        
    except Exception as e:
        logger.error(f"Error in combine_audio_with_metadata: {e}")
        return result

# Example usage
def main():
    """Example usage of the audio combination function"""
    input_dir = "output"  # Directory containing numbered .wav files
    output_file = "complete_audiobook.wav"
    
    print("🎵 Audio File Combiner")
    print("=" * 30)
    
    if not Path(input_dir).exists():
        print(f"❌ Directory '{input_dir}' does not exist")
        return
    
    # Combine with metadata
    result = combine_audio_with_metadata(input_dir, output_file)
    
    if result["success"]:
        print(f"✅ Successfully combined {result['files_processed']} files")
        print(f"📁 Output: {result['output_file']}")
        print(f"⏱️ Total duration: {result['total_duration']:.2f} seconds")
    else:
        print("❌ Failed to combine audio files")

if __name__ == "__main__":
    main()