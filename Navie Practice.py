# from sklearn import datasets
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.naive_bayes import MultinomialNB
#
# from  sklearn.pipeline import Pipeline
#
# Target = ["Sports","politics","Rowdy","Un Employers","Cricket","Computer","Cinima"]
# train = ["Kids are playing in ground They like cricket,coco,vollyball,rukby,chesss etc.games",
#          "Politician are speaking for votes in election for win",
#          "Rowdy are sporting for nation",
#          "Un employers are searching for jobs",
#          "Pubji is very intresting  in now a days ",
#          "Now a days computer is growing very fast",
#          "I have to act in movies \
#          Thala Ajith is leadind star in Tamil film industry"]
# vector = CountVectorizer()
# Count = vector.fit_transform(train)
# Tf = TfidfTransformer()
# New   = Tf.fit_transform(Count)
# # print(New.shape);print()
# # print(vector.vocabulary_);print()
# # print(vector.get_feature_names())
#
# Clf = MultinomialNB().fit(New,Target)
#
#
#
# # t = ["Python is programming langauge","Sachin is a cricket player"]
# text = ["I'm a job searching person","thala ajith is hero","Politian speach are marvelous they speaks not no use after winning election"]
# text1 = vector.transform(text)
# text2 =Tf.transform(text1)
# print(text2.shape)
# Predict = Clf.predict(text2)
# for x in Predict:
#         print(x)
#
# # from sklearn.metrics import accuracy_score
# # print("Accuracy is :",accuracy_score(text2,predict))


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

Target = ["self Introduction","Sports","Politics","Computer","Cinema"]
Data = ["My name is Amarnath , I have another name is  Kodali Amarnath , Ms. Sheela is my feature wife , My name is priya, i love my self ",
        'I love cricket ,chess many games i love , because of INDIA is soprt country, Sachin, Dhoni is players in cricket',
        'Politician have a lot of partys DMK ,ADMK, BJP, YJP, these are the i know the partyes of the Tamil-nadu .politician will not do as per the promicess.',
        "Present is computer is leding programming language . Python , Java script is secondary language, HTML and PHP is web design languages ",
        "Ajith , Vijay , Surya, Vishal , Vikram, is the hero of the Tamil Film Industry, We have a lot of heros are avaliable in Tamil film industry."]
Co = CountVectorizer()
Count = Co.fit_transform(Data)
count = TfidfTransformer()
Pre = count.fit_transform(Count)

Nomi = MultinomialNB()
Helo= Nomi.fit(Pre,Target)
print(Helo)

# Checking the Data
text = ["My name is Sheela","Jgan has startes in partys in Andhra_Pradesh","Im learning the Python Programming language"]
T1 = Co.transform(text)
T2 = count.transform(T1)
T3 = Nomi.predict(T2)
for x in T3:
    print(x)
from sklearn.metrics import classification_report
