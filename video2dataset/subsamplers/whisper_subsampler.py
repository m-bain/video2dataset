"""
Whisper subsampler - transcribes audio using the Whisper model from OAI using WhisperX API

code: https://github.com/m-bain/whisperX
"""
import os
import time
import tempfile

try:
    # import whisperx
    import faster_whisper
    import torch
except:  # pylint: disable=broad-except,bare-except
    pass

from .subsampler import Subsampler


class WhisperSubsampler(Subsampler):
    """
    Transcribes audio samples using the OAI Whisper Model via WhisperX API

    Params:
        model_name: https://github.com/guillaumekln/faster-whisper/blob/20d4e9418b5efb69ec5aa4819a39e3fb0e772a2a/faster_whisper/transcribe.py#LL90C1-L90C1
        batch_size: batch size used during inference (try to maximize this for perf)
        compute_type: accuracy/mem tradeoff (float16, float32, int8)
    """

    def __init__(
        self,
        model_name="large-v2",
        batch_size=16,
        beam_size=5,
        vad_filter=True,
        compute_type="float16",
        download_root=None,
        is_slurm_task=False,
    ):
        if is_slurm_task:
            global_rank = os.environ["GLOBAL_RANK"]
            if global_rank != 0:
                time.sleep(20)  # let master worker download model

            device, device_index = "cuda", int(os.environ["LOCAL_RANK"])
            while True:
                try:
                    self.model = faster_whisper.WhisperModel(
                        model_name,
                        device=device,
                        device_index=device_index,
                        compute_type=compute_type,
                        download_root=download_root,
                    )
                    print("model_loaded", os.environ["GLOBAL_RANK"], flush=True)
                    break
                except Exception as e:  # pylint: disable=(broad-except)
                    print(str(e), flush=True)
                    print(
                        "loading failed, retrying...",
                        os.environ["GLOBAL_RANK"],
                        flush=True,
                    )
                    continue
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = faster_whisper.WhisperModel(
                model_name,
                device=device,
                device_index=0,
                compute_type=compute_type,
                download_root=download_root,
            )

        self.model_name = model_name
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.vad_filter = vad_filter

    def __call__(self, streams, metadata=None):
        audio_bytes = streams.get("audio")
        for i, aud_bytes in enumerate(audio_bytes):
            # TODO: .m4a not always
            try:
                with tempfile.NamedTemporaryFile(suffix=".flac") as tmpfile:
                    tmpfile.write(aud_bytes)
                    tmpfile.flush()  # ensure all data is written
                    segments, info = self.model.transcribe(
                        tmpfile,
                        beam_size=self.beam_size,
                    )
                    result = []
                    for segment in segments:
                        result.append(
                            {
                                "start": segment.start,
                                "end": segment.end,
                                "text": segment.text,
                            }
                        )
                    
                    if len(result) > 0:
                        if "transcript" not in metadata[i]:
                            metadata[i]["transcript"] = []
                        metadata[i]["transcript"].append({
                            "segments": result,
                            "language": info.language,
                            "language_probability": info.language_probability,
                            "config": {
                                "model": self.model_name,
                                "batch_size": self.batch_size,
                                "beam_size": self.beam_size,
                                "vad_filter": self.vad_filter,
                                "engine": "faster_whisper",
                                "compute_type": self.compute_type,
                            },
                        })
            except Exception as err:  # pylint: disable=broad-except
                return [], metadata, str(err)

        return streams, metadata, None
