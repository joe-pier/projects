lista = ['facemill', 'slotdrill', 'ballendmill', 'ballendmill']


def toolchanger(tools, k, q):
    #sorry, i prefer to use my own IDE, this is my code
    lista=[]

    temp=-1
    for j in range(k,len(tools)):

        if tools[j]==q:
            temp=temp+1
            print('avanti di uno')
            break
        else:
            print('avanti di uno')
            temp = temp + 1
            if j+1 ==len(tools):
                for h in range(0,k):

                    if tools[h] == q:
                        break
                    else:
                        print('avanti di uno')
                        temp = temp + 1
    #print(f'avanti {temp}')


    temp_ = 0
    for i in reversed(range(-1,k)):

        if tools[i]==q:
            print('indietro di uno')
            temp_=temp_+1
            break
        else:
            temp_ = temp_ + 1
            print('indietro di uno')
            if i+1==0:
                for s in reversed(range(k,len(tools)-1)):
                    if tools[s]==q:
                        break
                    else:
                        print('indietro di uno')
                        temp_ = temp_ + 1


    #print(f'indietro {temp_}')
    lista.append(temp_)
    lista.append(temp)
    #print(lista)
    print(lista)
    return min(lista)



var = ['a',
       'b',
       'c',
       'd',
       'e',
       'f',
       'g',
       'h',
       'i',
       'm',
       'm',
       'y',
       'o',
       'p',
       'q',
       'r'
       's',
       't']



toolchanger(var, 15, 't')



