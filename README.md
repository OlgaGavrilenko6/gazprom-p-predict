# Разработка алгоритмов машинного обучения для прогнозирования поведения киберфизической системы

# Содержание
- [Приложения к работе](#extra)
- [Запуск модуля моделирования](#runM)
- [Запуск сервера](#runS)
- [Запуск клиента](#runC)

# Приложения к работе <a name="runM"></a>

Все приложения к работе находится в папке [папке extra](./extra).

| <div align="center">Название</div> | <div align="center">Файл</div> |                                                  <div align="center">Описание</div>                                                   |
|:----------------------------------:|:------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------:|
|        **Описание проекта**        |  [desc.pdf](./extra/desc.pdf)  |     Описание проекта, включающее в себя список используемых технологий и их аргументацию, а также архитектуру системы  и ее схему     |
|   **Математическое обоснование**   |    [mo.pdf](./extra/mo.pdf)    | Математическое обоснование с аргументацией применения статистического метода и включающее в себя математическое описание всех моделей |
|     **Протокол тестирования**      |    [pt.pdf](./extra/pt.pdf)    |          Протокол тестирования, влючающий функциональные требования, информацию об испытаниях и функциональное тестирование           |


# Запуск модуля моделирования <a name="runM"></a>
Вся работа по моделированию находится в папке [папке modeling](./modeling).

Для запуска моделирования потребуется установленная среда Jupyter Lab и Python версии 3.12.2.

После этого следует запустить ее командой:
```
jupyter notebook
```

Среда после выполнения команды откроется в браузере, в ней надо открыть [файл моделирования](./modeling/gazprom.ipynb).

После следует последовательно запустить все блоки, построенные модели сохранятся в [отдельной папке](./modeling/models).

# Запуск сервера <a name="runS"></a>

Вся серверная часть находится в папке [папке back](./back).

Исполняемый файл называется [app.py](./back/app.py).

Для запуска нужно установить фреймворк flask командой:
```
cd back
pip3 install flask
```

После из этой же папки можно запустить исполняемый файл командой:
```
python3.12 -m flask run
```

Сервер запустится по адресу `http://localhost:5000`.

### Запуск клиента <a name="runC"></a>

Вся серверная часть находится в папке [папке front](./front).
Клинент представляет собой простую HTML форму, поэтому для запуска достаточно открыть файл [index.html](./front/index.html) в браузере.