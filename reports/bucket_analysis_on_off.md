## Outcome Buckets — anchor_baug_on

| bucket       |   count |
|:-------------|--------:|
| Abstain      |    1204 |
| High-Quality |     107 |
| Low-Quality  |     778 |
| Mid-Quality  |     617 |

### Examples — anchor_baug_on — High-Quality
| question                                                   | final_answer                                                                                                                                                     |   f1 |   faithfulness |
|:-----------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|-----:|---------------:|
| which movie won the oscar best visual effects in 2021?     | The movie that won the Oscar for Best Visual Effects in 2021 is "Tenet" [CIT:30395ec2-4247-4a23-8003-37dcfdf1531c_2].                                            |    1 |              1 |
| what company in the dow jones is the best performer today? | Salesforce was the best performer in the Dow Jones Industrial Average today, surging 3.5% during afternoon trading [CIT:ad69d406-676b-4242-8984-452dc65ad48f_0]. |    1 |              1 |
| what's the name of nashville's hockey team?                | The name of Nashville's hockey team is the Nashville Predators. [CIT:164cce18-533e-4357-b884-91688f11abc8_2]                                                     |    1 |              1 |

### Examples — anchor_baug_on — Mid-Quality
| question                                                                                        | final_answer                                                                                                                                                                              |       f1 |   faithfulness |
|:------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------:|---------------:|
| how many 3-point attempts did steve nash average per game in seasons he made the 50-40-90 club? | Steve Nash averaged 5.0 three-point attempts per game during the seasons he made the 50-40-90 club (2005-06, 2007-08, 2008-09, and 2009-10) [CIT:7bb29eb4-12f9-45f9-bf8a-66832b3c8962_1]. | 0.222222 |              1 |
| what are the countries that are located in southern africa.                                     | The countries located in Southern Africa are Botswana, Lesotho, Namibia, South Africa, and Swaziland [CIT:99150c8b-556e-4800-b9e6-d33e6b5adfa5_1].                                        | 0.451613 |              1 |
| what's the cooling source of the koeberg nuclear power station?                                 | The cooling source of the Koeberg nuclear power station is seawater, which is sourced from the South Atlantic Benguela Current [CIT:74ec83d9-0ebc-4f5e-b738-7cffbf262e74_3].              | 0.333333 |              1 |

### Examples — anchor_baug_on — Low-Quality
| question                                                                               | final_answer                                                                                                                                                                                    |   f1 |   faithfulness |
|:---------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----:|---------------:|
| in 2004, which animated film was recognized with the best animated feature film oscar? | The animated film recognized with the Best Animated Feature Oscar in 2004 was "Wallace & Gromit: The Curse of the Were-Rabbit" [CIT:8163a6f0-3238-4a69-ba60-a4a06090bc6f_0].                    |    0 |            1   |
| what is the average gross for the top 3 pixar movies?                                  | The average gross for the top 3 Pixar movies, which are "Incredibles 2" ($1.243 billion), "Toy Story 4" ($1.073 billion), and "Toy Story 3" ($1.067 billion), is approximately $1.128 billion . |    0 |            0.6 |
| what is the most active volcano in the philippines?                                    | Mayon, along with Taal, Kanlaon, Bulusan, and Ragang, as identified by the Philippine Institute of Volcanology and Seismology (PHIVOLCS) [CIT:3c2e454d-10a1-40eb-9534-f9440ecc4c3b_2].          |    0 |            0.6 |

### Examples — anchor_baug_on — Abstain
| question                                                                                                         | final_answer   |   f1 |   faithfulness |
|:-----------------------------------------------------------------------------------------------------------------|:---------------|-----:|---------------:|
| what is a movie to feature a person who can create and control a device that can manipulate the laws of physics? | I don't know   |    0 |              0 |
| where did the ceo of salesforce previously work?                                                                 | I don't know   |    0 |              0 |
| on which date did sgml distribute dividends the first time                                                       | I don't know   |    0 |              0 |

## Outcome Buckets — anchor_baug_off

| bucket       |   count |
|:-------------|--------:|
| Abstain      |    1204 |
| High-Quality |     104 |
| Low-Quality  |     774 |
| Mid-Quality  |     624 |

### Examples — anchor_baug_off — High-Quality
| question                                                   | final_answer                                                                                                                                                      |   f1 |   faithfulness |
|:-----------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----:|---------------:|
| which movie won the oscar best visual effects in 2021?     | The movie that won the Oscar for Best Visual Effects in 2021 is "Tenet" [CIT:30395ec2-4247-4a23-8003-37dcfdf1531c_2].                                             |    1 |              1 |
| what company in the dow jones is the best performer today? | Today, Salesforce was the best performer in the Dow Jones Industrial Average, surging 3.5% during afternoon trading [CIT:ad69d406-676b-4242-8984-452dc65ad48f_0]. |    1 |              1 |
| what's the name of nashville's hockey team?                | The name of Nashville's hockey team is the Nashville Predators. [CIT:164cce18-533e-4357-b884-91688f11abc8_2]                                                      |    1 |              1 |

### Examples — anchor_baug_off — Mid-Quality
| question                                                         | final_answer                                                                                                                                                                 |       f1 |   faithfulness |
|:-----------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------:|---------------:|
| what are the countries that are located in southern africa.      | The countries located in Southern Africa are Botswana, Lesotho, Namibia, South Africa, and Swaziland [CIT:99150c8b-556e-4800-b9e6-d33e6b5adfa5_1].                           | 0.451613 |              1 |
| what's the cooling source of the koeberg nuclear power station?  | The cooling source of the Koeberg nuclear power station is seawater, which is sourced from the South Atlantic Benguela Current [CIT:74ec83d9-0ebc-4f5e-b738-7cffbf262e74_3]. | 0.333333 |              1 |
| what is the ex-dividend date of microsoft in the 1st qtr of 2024 | Microsoft's ex-dividend date in the 1st quarter of 2024 was Wednesday, February 14, 2024 [CIT:9b4ed019-baaf-4231-b2bc-2a6394b41821_4].                                       | 0.6      |              1 |

### Examples — anchor_baug_off — Low-Quality
| question                                                                                        | final_answer                                                                                                                                                                                                                                                                                   |       f1 |   faithfulness |
|:------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------:|---------------:|
| how many 3-point attempts did steve nash average per game in seasons he made the 50-40-90 club? | In the seasons Steve Nash made the 50-40-90 club (2005-06, 2007-08, 2008-09, and 2009-10), he averaged 5.0, 5.0, 5.0, and 5.0 three-point attempts per game, respectively, resulting in an average of 5.0 attempts per game across those seasons [CIT:7bb29eb4-12f9-45f9-bf8a-66832b3c8962_1]. | 0.136364 |              1 |
| in 2004, which animated film was recognized with the best animated feature film oscar?          | The animated film recognized with the Best Animated Feature Oscar in 2004 was "Wallace & Gromit: The Curse of the Were-Rabbit" [CIT:8163a6f0-3238-4a69-ba60-a4a06090bc6f_0].                                                                                                                   | 0        |              1 |
| what is the average gross for the top 3 pixar movies?                                           | The average gross for the top 3 Pixar movies, which are Incredibles 2 ($1.243 billion), Toy Story 4 ($1.073 billion), and Toy Story 3 ($1.067 billion), is approximately $1.128 billion [CIT:aaaab7df-4fa8-48fb-a665-92e69517402d_2].                                                          | 0        |              1 |

### Examples — anchor_baug_off — Abstain
| question                                                                                                         | final_answer   |   f1 |   faithfulness |
|:-----------------------------------------------------------------------------------------------------------------|:---------------|-----:|---------------:|
| what is a movie to feature a person who can create and control a device that can manipulate the laws of physics? | I don't know   |    0 |              0 |
| where did the ceo of salesforce previously work?                                                                 | I don't know   |    0 |              0 |
| on which date did sgml distribute dividends the first time                                                       | I don't know   |    0 |              0 |
