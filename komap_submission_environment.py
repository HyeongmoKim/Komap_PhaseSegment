"""제출 코드의 패키지 설치 또는 현재 버전 확인."""
import argparse
import importlib.metadata as metadata
import subprocess
import sys
PACKAGES = ['ultralytics==8.3.221', 'segmentation-models-pytorch==0.5.0', 'transformers==4.57.1', 'timm==1.0.30', 'huggingface-hub==0.36.2', 'numpy==2.1.3', 'scipy==1.16.3', 'opencv-python==4.14.0.94', 'Pillow==11.3.0', 'psutil==5.9.5', 'tqdm==4.67.3', 'pandas==2.2.3', 'sympy==1.14.0']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    if args.install:
        metadata.version('torch')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *PACKAGES])
    for name in ('torch', 'torchvision', 'ultralytics', 'segmentation-models-pytorch', 'transformers', 'timm', 'numpy', 'scipy', 'opencv-python'):
        try:
            print(name, metadata.version(name))
        except metadata.PackageNotFoundError:
            print(name, 'not installed')

if __name__ == '__main__':
    main()
