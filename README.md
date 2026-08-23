# 그림책 사진관

사진을 손그림 화풍으로 바꿔 책에 넣을 수 있게 저장하는 Streamlit 앱입니다.
여러 장을 한 번에 올리고, 첫 장으로 화풍을 맞춘 뒤, 낱장 또는 ZIP으로 내려받습니다.

**전부 앱 안에서 처리합니다.** API 키가 필요 없고, 사진이 외부로 전송되지 않습니다.

## Streamlit Cloud에 올리기

1. GitHub에 새 저장소를 만들고 이 폴더의 파일을 그대로 올립니다.
2. share.streamlit.io 에서 **New app** → 저장소와 브랜치를 고르고 Main file path에 `app.py`를 넣습니다.
3. Deploy를 누르면 몇 분 뒤 주소가 나옵니다. Secrets 설정은 필요 없습니다.

## 로컬에서 실행

```bash
pip install -r requirements.txt
streamlit run app.py

