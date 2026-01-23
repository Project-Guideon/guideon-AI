# guideon-AI

FastAPI + LangGraph service for Guideon.

## Setup

```bash
python --version #을 통해서 파이썬 버젼 확인!!  3.11이 아니면 3.11버젼 다운로드

#window
py -3.11 -m venv .guideon_AI
#mac 생성 :python3.11 -m venv .guideon_AI 
 
#가상환경이름=guideon_AI 

# Windows 활성화부분
.guideon_AI\Scripts\activate
# macOS/Linux로 활성화하는 방법 둘중 맞게 실행
#=>source .guideon_AI/bin/activate


pip install -r requirements.txt #설치!!
#안될시 pip upgrade하면 되는듯

#.env.example의 환경 변수 템플릿을 실제 .env로 갖고옴(window)
Windows: copy .env.example .env  
# mac: cp .env.example .env
