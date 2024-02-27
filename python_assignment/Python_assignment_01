#Assignment-1

######## author: Manosh Sur Choudhury ##############


import math
#function for the implementation of given series
def series_sum(a):
    sum=0
    for i in range(1,a):
        sum=sum+(i**2)/5
    return sum

#Creating a class for each cat
class CatA:
    Personality = 'Energetic'
    Colors = 'Red-brown'
    Friendliness = 'High'
    a1=7 #initial position based in friendliness
    b1=6 #initial position based in friendliness
    def __init__(self,posA,posB):
        self.posA=posA
        self.posB=posB


print('initial position of catA'+'('+str(series_sum(CatA.a1))+','+str(series_sum(CatA.b1))+')')

class CatB:
    Personality = 'Calm'
    Colors = 'British Blue'
    Friendliness = 'Medium'
    a2=5 #initial position based in friendliness
    b2=4 #initial position based in friendliness
    def __init__(self,posA,posB):
        self.posA=posA
        self.posB=posB
        
print('initial position of catB'+'('+str(series_sum(CatB.a2))+','+str(series_sum(CatB.b2))+')')

class CatC:
    Personality = 'Outgoing'
    Colors = 'Black'
    Friendliness = 'Low'
    a3=3 #initial position based in friendliness
    b3=2 #initial position based in friendliness
    def __init__(self,posA,posB):
        self.posA=posA
        self.posB=posB


print('initial position of catC'+'('+str(series_sum(CatC.a3))+','+str(series_sum(CatC.b3))+')')

print('to stay at same position give input 0, to come closer to origin give value in negative, to move away give value in positive')


#calculating changed position of the cats
posA1= series_sum(CatA.a1)+int(input('give value of a for position of catA: \n'))
posB1= series_sum(CatA.b1)+int(input('give value of b for position of catA: \n'))

postion_CATA= CatA(posA1, posB1) 



posA2= series_sum(CatB.a2)+int(input('give value of a for position of catB: \n'))
posB2= series_sum(CatB.b2)+int(input('give value of b for position of catB: \n'))

postion_CATB= CatB(posA2, posB2) 


posA3= series_sum(CatC.a3)+int(input('give value of a for position of catc: \n'))
posB3= series_sum(CatC.b3)+int(input('give value of b for position of catc: \n'))

postion_CATC= CatC(posA3, posB3) 

print('A'+'('+str(postion_CATA.posA)+','+str(postion_CATA.posB)+')')
print('B'+'('+str(postion_CATB.posA)+','+str(postion_CATB.posB)+')')
print('C'+'('+str(postion_CATC.posA)+','+str(postion_CATC.posB)+')')

#calculating Euclidean distance of each cat from mouse in origin
position_of_catA=(postion_CATA.posA,postion_CATA.posB)
position_of_catB=(postion_CATB.posA,postion_CATB.posB)
position_of_catC=(postion_CATC.posA,postion_CATC.posB)
position_of_mouse=(0,0)

d_1=math.dist(position_of_catA,position_of_mouse)
d_2=math.dist(position_of_catB,position_of_mouse)
d_3=math.dist(position_of_catC,position_of_mouse)


print('Cats distance from mouse respectively: '+str(d_1)+', '+str(d_2)+', '+str(d_3))

Z=[d_1,d_2,d_3]


#finding which mouse has the closest distance 
if min(Z) == d_1:
    print('catA is the winner') 
elif min(Z) == d_2:
    print('catB is the winner')
else:
    print('catC is the winner') 
    

input()