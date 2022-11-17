### 1. Configuration
`main.py`의 Argument들은 다음과 같습니다 :

- `--lr`: 모델의 학습율 (float)
- `--batch_size`: 배치 사이즈 (int)
- `--cpu`: cpu 사용 여부 (bool)
- `--save_dir`: 모델 체크포인트 및 로그 파일 저장 경로 (str)

# 2. Data
주식 데이터는 `marcap` 라이브러리를 통해 online 형태로 사용합니다.

# 3. Requirements
해당 코드를 실행 시키기 위해 필요한 모듈은 다음과 같습니다 :
`numpy, pandas, torch, tqdm, marcp, sklearn`.

# 5. References
[AttentionGRU](https://snu-primo.hosted.exlibrisgroup.com/primo-explore/fulldisplay?docid=82SNU_INST71820738270002591&context=L&vid=82SNU&lang=ko_KR&search_scope=ALL&adaptor=Local%20Search%20Engine&tab=all&query=any,contains,%EC%96%91%EC%A4%80%EC%97%B4)