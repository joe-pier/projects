import math

class BinaryTree():
    def __init__(self, d, u, n, p, K, rho):
        self.d = d
        self.u = u
        self.n = n-1
        self.p = p
        self.K = K
        self.rho=rho
        self.S = [[self.p]]
        self.CallTree = []

    def price_tree(self):

        for i in range(0, self.n):
            temp_ = []
            for k in self.S[-1]:
                temp = []
                for j in [self.u, self.d]:
                    var = round(float(k * j), 2)
                    temp.append(var)
                temp_ = temp_ + temp

            self.S.append(remove_duplicate(temp_))
        return self.S

    def print_tree(self, h):
        for n,i in enumerate(h):
            print('   '*(len(h)-n),end=' ')
            for k in i:
                print(k, end=' ')


            print((len(h)-(n+1))*' ',end='\n')



    def payoff_CALL(self):

        for i in self.S[1:]:
            i = list(i)
            for j in i.copy():
                i.remove(j)
                j = round(max(j-self.K , 0), 5)
                i.append(j)
            self.CallTree.append(i)
        return self.CallTree[-1]

    def probability(self):
        pi_= (math.exp(-self.rho * ((12/self.n)/12)) - self.d)/(self.u-self.d)
        return pi_

def remove_duplicate(x):
    return list(dict.fromkeys(x))





strike = 40
price = 100
d = 0.5
u = 2
n=2
rho=0.02
h=(12/n)/12

#istanzio un oggetto BinaryTree (funziona fino a 5 step)

call_option_1 = BinaryTree(d, u, n, price, strike, rho)

#costruisco l'albero dei prezzi

price_tree = call_option_1.price_tree()

#costruisco l'albero dei payoffs dell'opzione call

pay_off_call_tree = call_option_1.payoff_CALL()
pi_greco = call_option_1.probability()
print(pi_greco)

def func(rho, payoffs, h , prob):
    return math.exp(-rho*h/12)*(prob * payoffs[0] + (1-prob) * payoffs[1])

print(func(rho, pay_off_call_tree, h, pi_greco ))


'''
print('albero binomiale dei prezzi: ')
print(price_tree)
print('albero binomiale dei payoffs della call: ')
print(pay_off_call_tree)


print('\n')
call_option_1.print_tree(call_option_1.S)
'''
