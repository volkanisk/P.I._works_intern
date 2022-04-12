import numpy as np

data = np.genfromtxt("new_stats.csv", dtype= [("U20"),("U20"),("i8")], delimiter = ",")

country = np.asarray(data["f0"])
country = np.reshape(country,[len(country),1])

date= np.asarray(data["f1"])
date= np.reshape(date,[len(date),1])

vaccinations = np.asarray(data["f2"])
vaccinations= np.reshape(vaccinations,[len(vaccinations),1])


country_list= np.asarray(np.unique(country))

medians= np.zeros(((country_list.shape[0]),1))
iteration=0
for i in country_list:
    string = country_list[iteration]
    med = np.median(vaccinations[country == string])
    medians[iteration] = medians[iteration] + med
    iteration +=1

sorts = np.argsort(medians, axis=0)


print("The highest median is in", country_list[sorts[-1]])
print("The second highest median is in", country_list[sorts[-2]])
print("The third highest median is in", country_list[sorts[-3]])


