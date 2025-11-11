# vild_parser_student.py 

import torch
import torchaudio
import torchaudio.transforms as T
import os
import sys

"""
[Deprecated: 잘못된 utils 상대경로]
# utils 경로 추가
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from parser_utils import load_audio_file
from vild_utils import normalize_mel_shape
"""

# [변경] 프로젝트 루트/유틸/본 폴더 경로를 안전하게 추가
VILD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(VILD_DIR)
UTILS_DIR = os.path.join(PROJECT_ROOT, 'utils')
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)
if VILD_DIR not in sys.path:
    sys.path.append(VILD_DIR)

from parser_utils import load_audio_file
from vild_utils import normalize_mel_shape


class AudioParser:
    """
    Student 모델 학습용 오디오 파서

    주요 기능:
    - 오디오 로드 및 리샘플링
    - Mel-spectrogram 변환
    - 일정 길이의 segment로 분할 후 normalize
    - 각 segment는 [1, 64, 101] Tensor로 반환됨
    """

    def __init__(self, config):
        self.config = config
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.fft_size,
            hop_length=config.hop_length,
            n_mels=config.n_mels
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        self.resampler_cache = {}

        try:
            torchaudio.set_audio_backend("soundfile")
        except RuntimeError:
            pass

    # --- 함수 시그니처에서 max_segment 인자 제거 ---
    def load_and_segment(self, file_path):
        """
        오디오 파일을 mel-spectrogram으로 변환하고 segment 단위로 나눔
        Teacher와 동일하게, 항상 config에 정의된 max_segments 수만큼 반환합니다.

        Args:
            file_path (str): 오디오 파일 경로

        Returns:
            List[Tensor]: [1, 64, 101] 크기의 mel segment 텐서 리스트 (항상 max_segments 개수)
        """
        waveform = load_audio_file(file_path, self.config.sample_rate, self.resampler_cache)
        if waveform is None or waveform.numel() == 0:
            print(f"[Warning] Invalid waveform from: {file_path}")
            return []

        try:
            mel = self.mel_transform(waveform)
            mel_db = self.amplitude_to_db(mel)

            if mel_db.ndim != 3 or mel_db.shape[1] != 64:
                print(f"[Warning] Unexpected mel shape: {mel_db.shape} from {file_path}")
                return []

            _, _, total_time = mel_db.shape
            stride = self.config.segment_hop
            window = self.config.segment_length
            
            # --- Teacher와 동일한 max_segments 값을 config에서 가져옴 ---
            max_segments = getattr(self.config, "max_segments", 5)  # check_needed

            if total_time < window:
                print(f"[Warning] Mel too short for segmentation: {total_time} < {window} in {file_path}")
                return []

            segment_list = []
            for start in range(0, total_time - window + 1, stride):
                # --- 세그먼트 수가 max_segments에 도달하면 루프 중단 ---
                if len(segment_list) >= max_segments:
                    break
                
                segment = mel_db[:, :, start:start + window]
                normed = normalize_mel_shape(segment)
                if normed is not None:
                    segment_list.append(normed)
                else:
                    # --- 이 부분이 ---
                    print(f"[Skip] Segment normalize 실패 in {file_path}")

            if not segment_list:
                print(f"[Warning] No valid segment for: {file_path}")
                return []

            # --- Teacher 파서와 동일한 패딩 로직 추가 ---
            # 세그먼트 수가 최대치보다 적고 0개가 아니면, 마지막 세그먼트를 복사하여 개수를 맞춤
            if 0 < len(segment_list) < max_segments:
                last_valid = segment_list[-1]
                segment_list += [last_valid.clone() for _ in range(max_segments - len(segment_list))]

            return segment_list

        except Exception as e:
            print(f"[ERROR] Exception while parsing {file_path}: {e}")
            return []
    