# Data

## `panel_26_countries_2000_2020.xlsx` — main thesis panel

Balanced annual panel, 2000–2020, of international tourist arrivals in Algeria from 26 source countries (546 observations): Belgium, Brazil, Canada, China, Denmark, Egypt, France, Germany, Greece, Italy, Japan, Jordan, Kuwait, Mexico, Morocco, Netherlands, Norway, Portugal, Saudi Arabia, Spain, Sweden, Switzerland (`Swiss`), Tunisia, Turkey, UK, US.

| Column | Variable | Source |
|---|---|---|
| `year` | 2000–2020 | — |
| `id` | source country | — |
| `TA` | International tourist arrivals in Algeria from country *i* | Algerian Ministry of Tourism and Handicrafts |
| `GDPpc` | GDP per capita of country *i*, current US$ | World Bank WDI |
| `TCPIt` | Relative price index, Tunisia vs. country *i* (CPI Tunisia / CPI *i*) — competing destination | World Bank WDI |
| `TCPIm` | Relative price index, Morocco vs. country *i* — competing destination | World Bank WDI |
| `AC` | Accommodation capacity in Algeria (beds) | Ministry of Tourism and Handicrafts |
| `DIST` | Distance between Algiers and the capital of country *i*, km | distance.to |
| `TOP` | Trade openness between Algeria and country *i*: (X + M) / (GDP_Algeria + GDP_i) | IMF DOTS and World Bank WDI |
| `POP` | Population of country *i* | World Bank WDI |
| `INT` | Internet users in country *i*, % of population | World Bank WDI |


## `ind.xlsx` — transfer-learning panel

Annual data, 2000–2020, for Algeria, Egypt, Jordan, Morocco and Tunisia: `year`, `id`, `Number of tourists`, `Tourism receipts`, `Tourism expenditures` (World Bank WDI: ST.INT.ARVL, ST.INT.RCPT.CD, ST.INT.XPND.CD). Used by `src/01`, `02a`, `02` and `03`.
