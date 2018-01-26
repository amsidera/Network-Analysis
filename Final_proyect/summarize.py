"""
sumarize.py
"""
import collect
import classify
import cluster

def main():
    longitud, number = collect.main()
    average = cluster.main()
    positive, negative, ejemplo1, ejemplo2 = classify.main()
    f = open("summary.txt","w", encoding="utf-8")
    f.write("Number of users collected: %d\n" %longitud)
    f.write("Number of messages collected: %d\n" %number)
    f.write("Average number of users per community: %d\n" %average)
    f.write("Number of instances per class found: %d, %d\n" %(positive,negative))
    f.write("%s\n"%str(ejemplo1))
    f.write("%s\n"%str(ejemplo2))
    f.close()
    f2 = open("description.txt","w", encoding="utf-8")
    f2.write("Study of the impact caused by a tweet from a friend of DowJones in the stock market.\n")
    f2.write("I do a research on TOP 100 friends on Twitter of DowJones account.\n First, I downloaded the values ​​for SP500 for a couple of weeks each minute from finance.google.com.\n With each tweet from DowJones friends during that time, I saw the impact of that tweet on the stock market, calculating the value of the stock market at that time and subtract the value of the stock within 5 minutes of difference. Then I classified as positive if the subtraction is positive and say that type of tweet has a positive impact.\n Otherwise, if the subtraction is negative, the impact of the tweet had negative impact.\n")
    f2.write("In the classifier, with the training tweets I made a cluster of words with 10 means.\n The number of words in each cluster is one of the features that I have entered in my classifier. Another feature is the time each tweet was published.\n With this I have created a classifier that predicts the impact that the tweet of analysts will have on the stock market in SP500.\n")   
    f2.write("In the clusters, I have seen the number of communities that can exist. I have found that there are small communities.\n But at no time does it become a single community, like all stock analysts together in one cluster. Moreover I saw where each of the analysts friends was located. I have observed that it is not only the United States, the main country in the clusters, also we have United Kingdom, Iran or even Australia have some importance in these clusters.\n")
    f2.close()
if __name__ == '__main__':
    main()