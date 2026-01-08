# Alzheimer's detection template

## 개요
이 서비스는 뇌 MRI 한 장을 입력받아 치매 진행 단계를 4가지 클래스 중 하나로 분류하고,
Grad-CAM으로 모델이 주목한 위치를 시각화하는 데모입니다.

### 분류 클래스
- VeryMildDemented (매우 경미)
- MildDemented (경미)
- ModerateDemented (중등)
- NonDemented (비치매)

## 모델 특징과 커스텀 구성
- 기본 모델: ImageNet 사전학습 VGG16 백본
- 커스텀 분류기: VGG16 기본 classifier 제거 후, 3단 MLP head(512→2048→1024→num_classes)
- 백본 고정 옵션: 학습 시 VGG16 파라미터 고정 가능

## 추론/시각화 파이프라인
1. 입력 이미지(업로드 또는 로컬 경로) 로드
2. 전처리: 224x224 리사이즈, ImageNet mean/std 정규화
3. 모델 추론: softmax로 예측 클래스와 확률 계산
4. Grad-CAM 생성: 마지막 feature map 기반 중요도 히트맵 생성
5. 출력: Input / Grad-CAM / Overlay 3장 비교 표시

## Grad-CAM 색상 해석
- 빨간색/노란색: 모델이 강하게 주목한 영역
- 초록/파랑: 모델이 덜 주목한 영역
- 색이 진할수록 영향이 큼

## 참고/주의
- 본 서비스는 데모이며 의료 진단을 대체하지 않습니다.
- 모델 파일은 신뢰 가능한 경로의 파일만 사용하세요.
  - PyTorch `torch.load`는 임의 객체 로딩이 가능하므로, 외부에서 받은 모델 파일은 주의가 필요합니다.

## 실행방법
------------------------------

  1. 도커로 앱 실행

  - docker compose up -d (이미 올려둔 상태면 생략)

  2. 터널로 외부 노출

  - 작업 폴더에서 nohup ./cloudflared tunnel --url http://localhost:8501 --no-autoupdate --protocol http2 > cloudflared.log 2>&1 &
  - tail -n 20 cloudflared.log 로 발급된 https://...trycloudflare.com 주소 확인
  - 이 URL을 휴대폰/외부에서 접속

  3. 종료 시

  - pkill cloudflared (또는 kill <PID>)

  도커가 먼저 떠 있어야 8501에 서비스가 열리고, 터널이 그 포트를 바깥으로 중계합니다.


------------------------------

## 외부 접속(포트포워딩 불가 시) Cloudflare Tunnel 사용법
로컬 포트를 임시로 외부에 노출할 때 Cloudflare Tunnel을 사용합니다.

- 실행(백그라운드):
  ```bash
  cd /home/user1/hw/code/Hospital_image_processing
  nohup ./cloudflared tunnel --url http://localhost:8501 --no-autoupdate --protocol http2 > cloudflared.log 2>&1 &
  ```
  - 처음 실행 시 발급된 URL은 `cloudflared.log`에 기록됩니다.
  - URL 예시: `https://<something>.trycloudflare.com`

- 상태/URL 확인:
  ```bash
  tail -n 20 cloudflared.log
  ```
  또는
  ```bash
  ps aux | grep cloudflared
  ```

- 종료:
  ```bash
  kill <PID>          # ps aux에서 확인한 cloudflared PID
  # 또는
  pkill cloudflared
  ```

참고: quick tunnel은 프로세스가 종료되면 URL도 사라지며, 다시 실행하면 새 URL이 생성됩니다.
