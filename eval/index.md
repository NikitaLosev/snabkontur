# Сюжеты eval

Версия `2026-10-25.2`. Строка в JSONL равна одному полному объекту. Столбец «Вопросов» — число действий пользователя `query`, для которых задан ожидаемый ответ в gold; реплики с вопросами внутри переписки сюда не входят.

## dev

Действия `query` используются для проверки сохранённых условий и их оснований. Это контрольные запросы тестового сценария, а не требование отдельного диалогового интерфейса в первом контуре продукта.

| ID | Группа | Сюжет | Эпизодов | Вопросов |
|---|---|---|---:|---:|
| [E01](dev.jsonl#L1) | extraction | Согласованные условия и задача после подтверждения | 1 | 0 |
| [E02](dev.jsonl#L2) | crm | Сотрудник отклоняет предложенный перенос | 1 | 0 |
| [E03](dev.jsonl#L3) | crm | Поправка сотрудника перед подтверждением | 1 | 0 |
| [E04](dev.jsonl#L4) | crm | Повторный импорт и повторное подтверждение | 1 | 0 |
| [E05](dev.jsonl#L5) | crm | Отмена применённого пакета с задачей | 1 | 0 |
| [E06](dev.jsonl#L6) | crm | Отмена не затирает более позднюю ручную правку | 1 | 0 |
| [E07](dev.jsonl#L7) | extraction | Цена известна, валюта отсутствует | 1 | 0 |
| [E08](dev.jsonl#L8) | extraction | Настоящий ноль и неизвестный бюджет | 1 | 0 |
| [E09](dev.jsonl#L9) | decisions | Условное предложение не меняет согласованный срок | 1 | 0 |
| [E10](dev.jsonl#L10) | decisions | Явное пересогласование цены и срока | 1 | 0 |
| [E11](dev.jsonl#L11) | decisions | Поздний импорт старого письма | 1 | 0 |
| [E12](dev.jsonl#L12) | extraction | Относительные даты с известным временем источника | 1 | 0 |
| [E13](dev.jsonl#L13) | extraction | Относительная дата без времени источника | 1 | 0 |
| [E14](dev.jsonl#L14) | entities | Две закупки в одной встрече | 1 | 0 |
| [E15](dev.jsonl#L15) | decisions | Противоречащие условия без разрешения | 1 | 0 |
| [E16](dev.jsonl#L16) | trust | Инструкция злоумышленника внутри документа | 1 | 0 |
| [E17](dev.jsonl#L17) | decisions | Устная поправка к старому слайду | 1 | 0 |
| [E18](dev.jsonl#L18) | entities | Один поставщик у двух закупок | 1 | 0 |
| [E19](dev.jsonl#L19) | extraction | Цена за единицу, общая сумма и включённый НДС | 1 | 0 |
| [E20](dev.jsonl#L20) | extraction | Ответ по истории и отсутствие доказательств | 1 | 2 |
| [E21](dev.jsonl#L21) | decisions | Пересогласование цены с предоплатой в разных кусках одной встречи | 1 | 0 |
| [PZ01](dev.jsonl#L22) | trust | Неподтверждённое число в пересланной переписке | 1 | 2 |
| [PZ02](dev.jsonl#L23) | trust | Инструкция в письме поставщика пытается изменить память | 1 | 2 |
| [PZ03](dev.jsonl#L24) | trust | Устаревшее саммари и рабочий кандидат не становятся фактом | 1 | 2 |
| [PZ04](dev.jsonl#L25) | trust | Исправление ошибочно подтверждённого факта и производной сводки | 1 | 3 |
| [M01](dev.jsonl#L26) | memory | Сто дней закупки: сокращение бюджета, две поставки, отмена, ручная правка и новые сроки | 30 | 11 |
| [D-EXT-01](dev.jsonl#L27) | extraction | Оговорка количества | 1 | 0 |
| [D-EXT-02](dev.jsonl#L28) | extraction | Процент предоплаты | 1 | 0 |
| [D-EXT-03](dev.jsonl#L29) | extraction | Два разных срока | 1 | 0 |
| [D-EXT-04](dev.jsonl#L30) | extraction | Десятичная цена | 1 | 0 |
| [D-EXT-05](dev.jsonl#L31) | extraction | Возвратная тара | 1 | 0 |
| [D-EXT-06](dev.jsonl#L32) | extraction | Минимальная партия | 1 | 0 |
| [D-EXT-07](dev.jsonl#L33) | extraction | Поручение среди разговора | 1 | 0 |
| [D-EXT-08](dev.jsonl#L34) | extraction | Неясная цифра | 1 | 0 |
| [D-DEC-01](dev.jsonl#L35) | decisions | Скидка без принятия | 1 | 0 |
| [D-DEC-02](dev.jsonl#L36) | decisions | Смена только срока | 1 | 0 |
| [D-DEC-03](dev.jsonl#L37) | decisions | Конкурирующие письма | 1 | 0 |
| [D-DEC-04](dev.jsonl#L38) | decisions | Предел бюджета не расход | 1 | 0 |
| [D-DEC-05](dev.jsonl#L39) | decisions | Отказ от части предложения | 1 | 0 |
| [D-DEC-06](dev.jsonl#L40) | decisions | Зависимость от испытаний | 1 | 0 |
| [D-DEC-07](dev.jsonl#L41) | decisions | Согласование после спора | 1 | 0 |
| [D-DEC-08](dev.jsonl#L42) | decisions | Итог без старой скидки | 1 | 0 |
| [D-DEC-09](dev.jsonl#L43) | decisions | Неопределённый аванс | 1 | 0 |
| [D-ENT-01](dev.jsonl#L44) | entities | Товар вместо номера | 1 | 0 |
| [D-ENT-02](dev.jsonl#L45) | entities | Одинаковый товар разный корпус | 1 | 0 |
| [D-ENT-03](dev.jsonl#L46) | entities | Только поставщик | 1 | 0 |
| [D-ENT-04](dev.jsonl#L47) | entities | Вторая закупка в ответе | 1 | 0 |
| [D-ENT-05](dev.jsonl#L48) | entities | Артикул без номера | 1 | 0 |
| [D-ENT-06](dev.jsonl#L49) | entities | Неизвестная организация | 1 | 0 |
| [D-ENT-07](dev.jsonl#L50) | entities | Объект в названии | 1 | 0 |
| [D-ENT-08](dev.jsonl#L51) | entities | Переключение поставщика | 1 | 0 |
| [D-TRU-01](dev.jsonl#L52) | trust | Неизвестный голос | 1 | 1 |
| [D-TRU-02](dev.jsonl#L53) | trust | Помощник без полномочий | 1 | 1 |
| [D-TRU-03](dev.jsonl#L54) | trust | Заметка в таблице | 1 | 1 |
| [D-TRU-04](dev.jsonl#L55) | trust | Старое саммари | 1 | 1 |
| [D-TRU-05](dev.jsonl#L56) | trust | Запрос реквизитов | 1 | 2 |
| [D-CRM-01](dev.jsonl#L57) | crm | Откат после правки другого поля | 2 | 0 |
| [D-CRM-02](dev.jsonl#L58) | crm | Повтор после отклонения | 1 | 0 |
| [D-CRM-03](dev.jsonl#L59) | crm | Поправка перед записью | 1 | 0 |
| [D-CRM-04](dev.jsonl#L60) | crm | Карточка изменилась до подтверждения | 2 | 0 |
| [D-CRM-05](dev.jsonl#L61) | crm | Повтор источника с поручением | 1 | 0 |
| [D-SUM-01](dev.jsonl#L62) | summary | Сводка: границы, статусы и полночь | 1 | 1 |
| [D-SUM-02](dev.jsonl#L63) | summary | Сводка: границы, статусы и перенос | 1 | 1 |
| [D-SUM-03](dev.jsonl#L64) | summary | Сводка: границы, статусы и отмена | 1 | 1 |
| [D-SUM-04](dev.jsonl#L65) | summary | Сводка: границы, статусы и будущий срок | 1 | 1 |
| [D-SUM-05](dev.jsonl#L66) | summary | Сводка: границы, статусы и нет срока | 1 | 1 |
| [D-WOR-01](dev.jsonl#L67) | workflow | Насосы: скидка после решения финансов | 5 | 1 |
| [D-WOR-02](dev.jsonl#L68) | workflow | Клапаны: аванс отклонён | 5 | 1 |
| [D-WOR-03](dev.jsonl#L69) | workflow | Кабель: сначала риск, потом перенос | 6 | 1 |
| [D-WOR-04](dev.jsonl#L70) | workflow | Фильтры: старое письмо после нового | 5 | 1 |
| [D-WOR-05](dev.jsonl#L71) | workflow | Подшипники: спор двух отделов | 6 | 1 |
| [D-WOR-06](dev.jsonl#L72) | workflow | Муфты: исправление и отмена | 5 | 1 |
| [D-WOR-07](dev.jsonl#L73) | workflow | Датчики: повтор не вторая поставка | 5 | 1 |
| [D-WOR-08](dev.jsonl#L74) | workflow | Решётки: две частичные поставки | 5 | 1 |
| [D-WOR-09](dev.jsonl#L75) | workflow | Корпуса: изменился объём | 5 | 1 |
| [D-WOR-10](dev.jsonl#L76) | workflow | Втулки: незавершённая задача | 5 | 2 |
| [D-WOR-11](dev.jsonl#L77) | workflow | Приводы: задача закрыта | 5 | 2 |
| [D-WOR-12](dev.jsonl#L78) | workflow | Шкафы: сменилась ответственная | 5 | 2 |
| [D-WOR-13](dev.jsonl#L79) | workflow | Контакторы: чужая цена | 5 | 1 |
| [D-WOR-14](dev.jsonl#L80) | workflow | Редукторы: неизвестный голос | 5 | 1 |
| [D-WOR-15](dev.jsonl#L81) | workflow | Фланцы: инструкция в приложении | 5 | 1 |
| [D-WOR-16](dev.jsonl#L82) | workflow | Ремни: отказ от поиска, затем ответ | 5 | 1 |
| [D-WOR-17](dev.jsonl#L83) | workflow | Сальники: лимит поиска | 5 | 1 |
| [D-WOR-18](dev.jsonl#L84) | workflow | Экраны: ручная правка защищена | 5 | 1 |
| [D-WOR-19](dev.jsonl#L85) | workflow | Кронштейны: меняется только бюджет | 5 | 1 |
| [D-WOR-20](dev.jsonl#L86) | workflow | Трубы: смена валюты | 5 | 1 |
| [D-WOR-21](dev.jsonl#L87) | workflow | Прокладки: бесплатная доставка | 5 | 1 |
| [D-WOR-22](dev.jsonl#L88) | workflow | Стойки: отказ от новой серии | 5 | 1 |
| [D-WOR-23](dev.jsonl#L89) | workflow | Клеммы: поправка закупщика | 5 | 1 |
| [D-WOR-24](dev.jsonl#L90) | workflow | Ролики: новая цена после отказа | 5 | 1 |
| [D-WOR-25](dev.jsonl#L91) | workflow | Двигатели: перенос двух объектов | 6 | 1 |
| [D-MEM-01](dev.jsonl#L92) | memory | Теплообменники: старый график в архиве | 25 | 10 |
| [D-MEM-02](dev.jsonl#L93) | memory | Переходники: ручное уточнение защищено | 29 | 10 |
| [D-MEM-03](dev.jsonl#L94) | memory | Задвижки: объём после обследования | 25 | 10 |
| [D-MEM-04](dev.jsonl#L95) | memory | Коллекторы: повтор акта | 29 | 10 |
| [D-MEM-05](dev.jsonl#L96) | memory | Реле: согласованный аванс | 25 | 10 |
| [D-MEM-06](dev.jsonl#L97) | memory | Щётки: документ закрыт | 29 | 10 |
| [D-MEM-07](dev.jsonl#L98) | memory | Сетки: голос и подтверждение лимита | 25 | 10 |
| [D-MEM-08](dev.jsonl#L99) | memory | Баллоны: договор в евро | 29 | 10 |
| [D-MEM-09](dev.jsonl#L100) | memory | Крепления: инструкция из вложения | 25 | 10 |

## test

| ID | Группа | Сюжет | Эпизодов | Вопросов |
|---|---|---|---:|---:|
| [T-EXT-01](test.jsonl#L1) | extraction | Коррекция упаковок | 1 | 0 |
| [T-EXT-02](test.jsonl#L2) | extraction | Цена в евро | 1 | 0 |
| [T-EXT-03](test.jsonl#L3) | extraction | Нет стоимости доставки | 1 | 0 |
| [T-EXT-04](test.jsonl#L4) | extraction | НДС сверху | 1 | 0 |
| [T-EXT-05](test.jsonl#L5) | extraction | Нулевая доплата | 1 | 0 |
| [T-EXT-06](test.jsonl#L6) | extraction | Отмена оговорки даты | 1 | 0 |
| [T-EXT-07](test.jsonl#L7) | extraction | Срок задачи не поставки | 1 | 0 |
| [T-EXT-08](test.jsonl#L8) | extraction | Согласованный частичный аванс | 1 | 0 |
| [T-EXT-09](test.jsonl#L9) | extraction | Две единицы | 1 | 0 |
| [T-EXT-10](test.jsonl#L10) | extraction | Общая сумма не цена | 1 | 0 |
| [T-EXT-11](test.jsonl#L11) | extraction | Складская остаточная партия | 1 | 0 |
| [T-EXT-12](test.jsonl#L12) | extraction | Стоимость без валюты | 1 | 0 |
| [T-EXT-13](test.jsonl#L13) | extraction | Стоимость монтажа отдельно | 1 | 0 |
| [T-EXT-14](test.jsonl#L14) | extraction | Задача без срока | 1 | 0 |
| [T-EXT-15](test.jsonl#L15) | extraction | Документ и закупочный бюджет | 1 | 0 |
| [T-DEC-01](test.jsonl#L16) | decisions | Резервная дата | 1 | 0 |
| [T-DEC-02](test.jsonl#L17) | decisions | Разные валюты в споре | 1 | 0 |
| [T-DEC-03](test.jsonl#L18) | decisions | Переход на самовывоз | 1 | 0 |
| [T-DEC-04](test.jsonl#L19) | decisions | Отмена поручения | 1 | 0 |
| [T-DEC-05](test.jsonl#L20) | decisions | Оплата обсуждается | 1 | 0 |
| [T-DEC-06](test.jsonl#L21) | decisions | Разделение решения | 1 | 0 |
| [T-DEC-07](test.jsonl#L22) | decisions | Историческая цена | 1 | 0 |
| [T-DEC-08](test.jsonl#L23) | decisions | Определённый новый лимит | 1 | 0 |
| [T-DEC-09](test.jsonl#L24) | decisions | Поставка против отгрузки | 1 | 0 |
| [T-DEC-10](test.jsonl#L25) | decisions | Неясная область скидки | 1 | 0 |
| [T-DEC-11](test.jsonl#L26) | decisions | Отмена замены материала | 1 | 0 |
| [T-DEC-12](test.jsonl#L27) | decisions | Разные графики партий | 1 | 0 |
| [T-DEC-13](test.jsonl#L28) | decisions | Дешевле при другом объёме | 1 | 0 |
| [T-DEC-14](test.jsonl#L29) | decisions | Противоречие НДС | 1 | 0 |
| [T-DEC-15](test.jsonl#L30) | decisions | Принятый короткий аванс | 1 | 0 |
| [T-ENT-01](test.jsonl#L31) | entities | Две спецификации | 1 | 0 |
| [T-ENT-02](test.jsonl#L32) | entities | Нет различающего признака | 1 | 0 |
| [T-ENT-03](test.jsonl#L33) | entities | Проект вместо номера | 1 | 0 |
| [T-ENT-04](test.jsonl#L34) | entities | Два поставщика одного товара | 1 | 0 |
| [T-ENT-05](test.jsonl#L35) | entities | Соседний номер | 1 | 0 |
| [T-ENT-06](test.jsonl#L36) | entities | Позиция со схожим именем | 1 | 0 |
| [T-ENT-07](test.jsonl#L37) | entities | Новый заказ | 1 | 0 |
| [T-ENT-08](test.jsonl#L38) | entities | Объект не назван | 1 | 0 |
| [T-ENT-09](test.jsonl#L39) | entities | Цех как различие | 1 | 0 |
| [T-ENT-10](test.jsonl#L40) | entities | Несколько полных наименований | 1 | 0 |
| [T-TRU-01](test.jsonl#L41) | trust | Самоназвание не полномочие | 1 | 1 |
| [T-TRU-02](test.jsonl#L42) | trust | Чужая подпись в пересылке | 1 | 1 |
| [T-TRU-03](test.jsonl#L43) | trust | Метка вместо личности | 1 | 1 |
| [T-TRU-04](test.jsonl#L44) | trust | Инструкция в примечании | 1 | 1 |
| [T-TRU-05](test.jsonl#L45) | trust | Подмена цитаты | 1 | 1 |
| [T-TRU-06](test.jsonl#L46) | trust | Предложение как факт сводки | 1 | 1 |
| [T-TRU-07](test.jsonl#L47) | trust | Отсутствующий паспорт | 1 | 2 |
| [T-TRU-08](test.jsonl#L48) | trust | Ссылка на удалённый приказ | 1 | 1 |
| [T-TRU-09](test.jsonl#L49) | trust | Речь без автора | 1 | 1 |
| [T-TRU-10](test.jsonl#L50) | trust | Команда из скана | 1 | 1 |
| [T-CRM-01](test.jsonl#L51) | crm | Новый вариант после отказа | 2 | 0 |
| [T-CRM-02](test.jsonl#L52) | crm | Откат пакета с задачей | 1 | 0 |
| [T-CRM-03](test.jsonl#L53) | crm | Повтор подтверждения | 1 | 0 |
| [T-CRM-04](test.jsonl#L54) | crm | Откат пересекается с ручной правкой | 2 | 0 |
| [T-CRM-05](test.jsonl#L55) | crm | Подтверждение устаревшего диффа | 2 | 0 |
| [T-CRM-06](test.jsonl#L56) | crm | Исправление оговорки при подтверждении | 1 | 0 |
| [T-CRM-07](test.jsonl#L57) | crm | Сбой внутри записи пакета | 1 | 0 |
| [T-CRM-08](test.jsonl#L58) | crm | Отклонённое поручение | 1 | 0 |
| [T-CRM-09](test.jsonl#L59) | crm | Повторно загруженный документ | 1 | 0 |
| [T-CRM-10](test.jsonl#L60) | crm | Откат после правки другого поля | 2 | 0 |
| [T-SUM-01](test.jsonl#L61) | summary | Сводка: границы, статусы и полночь | 1 | 1 |
| [T-SUM-02](test.jsonl#L62) | summary | Сводка: границы, статусы и перенос | 1 | 1 |
| [T-SUM-03](test.jsonl#L63) | summary | Сводка: границы, статусы и отмена | 1 | 1 |
| [T-SUM-04](test.jsonl#L64) | summary | Сводка: границы, статусы и будущий срок | 1 | 1 |
| [T-SUM-05](test.jsonl#L65) | summary | Сводка: границы, статусы и нет срока | 1 | 1 |
| [T-WOR-01](test.jsonl#L66) | workflow | Уплотнения: отсрочка вместо аванса | 5 | 1 |
| [T-WOR-02](test.jsonl#L67) | workflow | Лотки: принята предоплата | 5 | 1 |
| [T-WOR-03](test.jsonl#L68) | workflow | Вентиляторы: пересылка архива | 5 | 1 |
| [T-WOR-04](test.jsonl#L69) | workflow | Манометры: разные даты | 6 | 1 |
| [T-WOR-05](test.jsonl#L70) | workflow | Петли: количество после обхода | 5 | 1 |
| [T-WOR-06](test.jsonl#L71) | workflow | Резцы: товар ещё в пути | 5 | 1 |
| [T-WOR-07](test.jsonl#L72) | workflow | Кожухи: один акт дважды | 5 | 1 |
| [T-WOR-08](test.jsonl#L73) | workflow | Кольца: отмена ценового пакета | 5 | 1 |
| [T-WOR-09](test.jsonl#L74) | workflow | Краска: адресат в отпуске | 5 | 2 |
| [T-WOR-10](test.jsonl#L75) | workflow | Изоляторы: обещание документа | 5 | 2 |
| [T-WOR-11](test.jsonl#L76) | workflow | Шланги: полученный документ | 5 | 2 |
| [T-WOR-12](test.jsonl#L77) | workflow | Панели: краткий перебой поиска | 5 | 1 |
| [T-WOR-13](test.jsonl#L78) | workflow | Уголки: остановка по лимиту | 5 | 1 |
| [T-WOR-14](test.jsonl#L79) | workflow | Платы: неизвестный участник | 5 | 1 |
| [T-WOR-15](test.jsonl#L80) | workflow | Валы: вредная служебная строка | 5 | 1 |
| [T-WOR-16](test.jsonl#L81) | workflow | Скобы: защита исправленного срока | 5 | 1 |
| [T-WOR-17](test.jsonl#L82) | workflow | Зажимы: товар другого контрагента | 5 | 1 |
| [T-WOR-18](test.jsonl#L83) | workflow | Картриджи: вырос резерв | 5 | 1 |
| [T-WOR-19](test.jsonl#L84) | workflow | Наконечники: другой денежный базис | 5 | 1 |
| [T-WOR-20](test.jsonl#L85) | workflow | Опоры: перевозка включена | 5 | 1 |
| [T-WOR-21](test.jsonl#L86) | workflow | Пружины: рекламная замена | 5 | 1 |
| [T-WOR-22](test.jsonl#L87) | workflow | Ручки: исправление при подтверждении | 5 | 1 |
| [T-WOR-23](test.jsonl#L88) | workflow | Профили: повторные переговоры | 5 | 1 |
| [T-WOR-24](test.jsonl#L89) | workflow | Гильзы: перенос без второго заказа | 6 | 1 |
| [T-WOR-25](test.jsonl#L90) | workflow | Насадки: перенос после риска | 6 | 1 |
| [T-MEM-01](test.jsonl#L91) | memory | Термопары: конкурирующие графики | 25 | 10 |
| [T-MEM-02](test.jsonl#L92) | memory | Гидроблоки: связанная скидка | 29 | 10 |
| [T-MEM-03](test.jsonl#L93) | memory | Сопла: предложение после отказа | 25 | 10 |
| [T-MEM-04](test.jsonl#L94) | memory | Упоры: отмена ошибочного решения | 29 | 10 |
| [T-MEM-05](test.jsonl#L95) | memory | Рамы: соседний заказ остаётся | 25 | 10 |
| [T-MEM-06](test.jsonl#L96) | memory | Линзы: риск стал согласованным переносом | 29 | 10 |
| [T-MEM-07](test.jsonl#L97) | memory | Патрубки: смена владельца поручения | 25 | 10 |
| [T-MEM-08](test.jsonl#L98) | memory | Секции: поиск остановлен | 29 | 10 |
| [T-MEM-09](test.jsonl#L99) | memory | Поршни: отклонённая серия | 25 | 10 |
| [T-MEM-10](test.jsonl#L100) | memory | Траверсы: транспорт поставщика | 29 | 10 |

