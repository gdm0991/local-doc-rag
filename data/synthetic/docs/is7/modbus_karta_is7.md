# Карта регистров Modbus TCP стенда ИС-7

> Синтетический документ для демонстрации local-doc-rag. Стенд, оборудование и значения вымышлены.

Адрес контроллера 192.168.12.50, порт 502, Unit ID 12. Значения аналоговых параметров — REAL (2 регистра, порядок слов big-endian).

| Тег | Параметр | Область | Адрес | Тип |
|---|---|---|---|---|
| N_ROT | Частота вращения вала | Input Registers | 30001 | REAL |
| M_TORQ | Крутящий момент на входном валу | Input Registers | 30003 | REAL |
| P_OIL | Давление масла на выходе насоса | Input Registers | 30005 | REAL |
| V_BRG | Виброскорость опоры подшипника | Input Registers | 30007 | REAL |
| I_MOT | Ток двигателя привода | Input Registers | 30009 | REAL |
| Q_OIL | Расход масла | Input Registers | 30011 | REAL |
| T_GEAR | Температура масла в картере редуктора | Input Registers | 30013 | REAL |
| T_OIL | Температура масла в баке | Input Registers | 30015 | REAL |
| N_ROT_ALM | Аварийная уставка: частота вращения вала | Holding Registers | 40001 | REAL |
| M_TORQ_ALM | Аварийная уставка: крутящий момент на входном валу | Holding Registers | 40003 | REAL |
| T_GEAR_ALM | Аварийная уставка: температура масла в картере редуктора | Holding Registers | 40005 | REAL |
| P_OIL_ALM | Аварийная уставка: давление масла на выходе насоса | Holding Registers | 40007 | REAL |
| T_OIL_ALM | Аварийная уставка: температура масла в баке | Holding Registers | 40009 | REAL |
| V_BRG_ALM | Аварийная уставка: виброскорость опоры подшипника | Holding Registers | 40011 | REAL |
| I_MOT_ALM | Аварийная уставка: ток двигателя привода | Holding Registers | 40013 | REAL |
| Q_OIL_ALM | Аварийная уставка: расход масла | Holding Registers | 40015 | REAL |
