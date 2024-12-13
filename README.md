# A3C 강화 학습을 활용한 주식 거래 자동화

이 프로젝트는 **Asynchronous Advantage Actor-Critic (A3C)** 강화 학습 모델을 사용하여 주식 거래를 자동화하는 방법을 보여줍니다. 
모델은 과거 주식 데이터와 다양한 기술 지표를 기반으로 매수, 매도, 또는 보유 결정을 내립니다.

---

## 목차
- [소개](#소개)
- [기능](#기능)
- [설치 방법](#설치-방법)
- [Python 스타일 가이드](#python-스타일-가이드)

---

## 소개
자동화된 주식 거래는 시장 동향, 가격 변동, 경제 지표 등을 종합적으로 고려해야 하는 복잡한 작업입니다. 이 프로젝트에서는 강화 학습 알고리즘 중 하나인 **A3C 모델**을 사용하여 주식 거래에서 기대 수익을 극대화하는 정책을 학습합니다.

모델은 주식 가격, 현금 잔고, 그리고 기술 지표(이동 평균선, 거래량 평균선 등)를 관찰하여 **최적의 매매 전략**을 학습합니다.

---

## 기능
- **강화 학습 기반 거래**: A3C 알고리즘을 사용하여 거래 전략을 학습합니다.
- **기술 지표 포함**: Simple Moving Average(SMA), Volume Moving Average(VMA)와 같은 지표를 관찰 공간에 포함.
- **실시간 거래 시뮬레이션**: 모델의 예측을 기반으로 실시간으로 거래를 시뮬레이션.
- **성과 시각화**: 계좌 가치와 주식 가격 변동, 매수/매도 시점을 그래프로 표시.

---

## 설치 방법 <br>

파이썬 버전 3.11.4 사용

### 1. 레포지토리 클론
```bash
git clone https://github.com/yourusername/ICTRLT.git
cd ICTRLT
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

3. **Google Drive Integration**: <br>
This project uses Google Colab for training. Ensure you have Google Drive mounted in your Colab environment.

## Python 스타일 가이드
이 프로젝트는 **PEP 8 스타일 가이드를 준수**합니다.  
코드를 작성하거나 수정할 때는 해당 규칙을 참고하세요.
