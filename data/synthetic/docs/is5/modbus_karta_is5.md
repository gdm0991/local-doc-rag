# Карта регистров Modbus TCP стенда ИС-5

> Синтетический документ для демонстрации local-doc-rag. Стенд, оборудование и значения вымышлены.

Адрес контроллера 192.168.11.50, порт 502, Unit ID 15. Значения аналоговых параметров — REAL (2 регистра, порядок слов big-endian).

| Тег | Параметр | Область | Адрес | Тип |
|---|---|---|---|---|
| P_FUEL | Давление топлива на входе регулятора | Input Registers | 30001 | REAL |
| Q_FUEL | Расход топлива | Input Registers | 30003 | REAL |
| N_ROT | Частота вращения вала | Input Registers | 30005 | REAL |
| P_OIL | Давление масла на выходе насоса | Input Registers | 30007 | REAL |
| I_MOT | Ток двигателя привода | Input Registers | 30009 | REAL |
| V_BRG | Виброскорость опоры подшипника | Input Registers | 30011 | REAL |
| T_FUEL | Температура топлива | Input Registers | 30013 | REAL |
| P_FUEL_ALM | Аварийная уставка: давление топлива на входе регулятора | Holding Registers | 40001 | REAL |
| Q_FUEL_ALM | Аварийная уставка: расход топлива | Holding Registers | 40003 | REAL |
| T_FUEL_ALM | Аварийная уставка: температура топлива | Holding Registers | 40005 | REAL |
| N_ROT_ALM | Аварийная уставка: частота вращения вала | Holding Registers | 40007 | REAL |
| P_OIL_ALM | Аварийная уставка: давление масла на выходе насоса | Holding Registers | 40009 | REAL |
| I_MOT_ALM | Аварийная уставка: ток двигателя привода | Holding Registers | 40011 | REAL |
| V_BRG_ALM | Аварийная уставка: виброскорость опоры подшипника | Holding Registers | 40013 | REAL |
