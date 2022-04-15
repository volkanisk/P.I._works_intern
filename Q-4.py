#Volkan Işık  15 April 2022
import numpy as np


data = np.genfromtxt("country_vaccination_stats.csv", dtype= [("U20"),("U20"),("i8")] , skip_header = 1 ,
                     delimiter = ",",  usecols= [0,1,2], filling_values= "999999")

country= np.asarray(data["f0"])
country= np.reshape(country,[len(country),1])

date= np.asarray(data["f1"])
date= np.reshape(date,[len(date),1])

vaccinations = np.asarray(data["f2"])
vaccinations= np.reshape(vaccinations,[len(vaccinations),1])



for i in range(country.shape[0]):
    if vaccinations[i] == 999999:
        min = np.min(vaccinations[country == country[i]])
        if min != 999999:
            vaccinations[i] = min
        else:
            vaccinations[i] = 0

stack= np.hstack([country,date,vaccinations])

np.savetxt("new_stats.csv",stack, fmt =  "%s",delimiter=",")

