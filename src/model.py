import numpy as np
import pandas as pd
import config

class CohortComponentModel:
    def __init__(self, base_pop, migration=None):
        self.base_pop = base_pop.copy()
        self.max_age = base_pop.index.max()
        if migration is None:
            self.migration = {'male': np.zeros(self.max_age+1),
                              'female': np.zeros(self.max_age+1)}
        else:
            self.migration = migration

    def step(self, pop_male, pop_female, asfr, surv_male, surv_female):
        """单步预测：输入当前人口、当年ASFR和存活率，返回下一年人口"""
        new_male = np.zeros(self.max_age+1)
        new_female = np.zeros(self.max_age+1)

        # 存活过程
        new_male[1:] = pop_male[:-1] * surv_male[:-1]
        new_female[1:] = pop_female[:-1] * surv_female[:-1]
        new_male[-1] += pop_male[-1] * surv_male[-1]
        new_female[-1] += pop_female[-1] * surv_female[-1]

        # 出生
        women = pop_female[config.FERTILE_AGE_START:config.FERTILE_AGE_END+1]
        births = np.sum(women * asfr)
        male_ratio = config.SEX_RATIO_BIRTH / (100 + config.SEX_RATIO_BIRTH)
        new_male[0] = births * male_ratio * surv_male[0]
        new_female[0] = births * (1 - male_ratio) * surv_female[0]

        # 迁移
        new_male += self.migration['male']
        new_female += self.migration['female']

        return new_male, new_female