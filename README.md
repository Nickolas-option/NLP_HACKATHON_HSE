# AI Assistant Hack: Python

## Запуск

### В локальной среде

```bash
pip install -r requirements.txt
```
0. создать .env файлик в корне и положить креды от YaGPT_API
```
YANDEX_GPT_API_KEY=
YANDEX_GPT_FOLDER_ID=
```
2. положить tasks и solutions в data/raw
3. запустить строчку из preprocess.py чтобы склеить tasks и solutions файлы: 
merge_task_to_solution(path_to_solutions="./data/raw/test/solutions.xlsx",
                           path_to_tasks="./data/raw/test/tasks.xlsx",
                           save_to="./data/processed/test/solutions.xlsx")
4. запустить tuning.py
                           
5. затем запустить файл main.py


