# `python -m imgsearch`로 패키지를 모듈 실행할 때의 진입점.
# 실제 부트스트랩 로직은 app.main()에 있고, 여기서는 그 종료 코드를 프로세스에 넘긴다.
from imgsearch.app import main

if __name__ == "__main__":
    raise SystemExit(main())
