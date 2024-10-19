# AI Assistant Hack: Python

## Запуск

### В локальной среде

```bash
pip install -r requirements.txt
```
1. положить tasks и solutions в data/raw
2. запустить строчку из preprocess.py чтобы склеить tasks и solutions файлы: 
merge_task_to_solution(path_to_solutions="./data/raw/test/solutions.xlsx",
                           path_to_tasks="./data/raw/test/tasks.xlsx",
                           save_to="./data/processed/test/solutions.xlsx")
3. запустить tuning.py
                           
4. затем запустить файл main.py


