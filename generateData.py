# Reference: S. Lee/ Journal of Econometrics 178 (2014) 398-413
# Generate the data needed for bootstrap
# The model is based on equation 7.12 of Lee (2014)
import numpy as np


# n number of samples, rho correlation between 1st and 2nd stages
# delta is the degree of misspecification, gamma1 coefficient of z1
# beta0 coefficition of the endogenous variable
def genData(n=200,delta=0.0,gamma1=0.25,beta0 = 0.0):
	exp1 = np.exp(1)
	exp2 = np.exp(2)
	rhoeu = 4.6223
	rho = 0.99
# generate uncorrelated instruments
	instruments = np.random.multivariate_normal([0,0],[[1,0],[0,1]],n)

# generate correlated lognormal errors
	errors =  np.random.multivariate_normal([0,0],[[1,rho],[rho,1]],n)
	lognerrors = np.exp(errors) # make them mean zero

# generate the variables in eqn 7.12
	z1 = instruments[:,0]
	z02 = instruments[:,1]
	eps =  lognerrors[:,0]-np.exp(0.5)
	u = lognerrors[:,1]-np.exp(0.5)
	gamma2 = -(delta*rhoeu)/(exp2-exp1+delta**2)
	z2 = z02 + delta*eps/((exp1-1)*exp1)
	x = z1*gamma1+z2*gamma2+u
	y = x*beta0+eps
	#beta0= (1+delta**2/((exp1-1)*exp1))*gamma2+delta/((exp1-1)*exp1)*rho
	data = np.array([y,x,z1,z2]).T
	return data


def getWildx(data,rade): # data has to be n by 4 array
	size = data.shape[0]
	x = np.matrix(np.array(data[:,1])).T
	W = np.matrix(np.array([data[:,2], data[:,3]]).T)
	ols = (W.T*W).I*W.T*x
	res = x - W*ols
	#binom = np.random.binomial(1,0.5,size)
	rade = np.matrix(rade).T # generate rademarcher dist
	fres =  np.multiply(rade,res) # transformed residues
	wildx = np.array(W*ols + fres)
	return wildx
	
def getIVEst(data): # data has to be n by 4 array
	y = np.matrix(np.array(data[:,0])).T
	x = np.matrix(np.array(data[:,1])).T
	W = np.matrix(np.array([data[:,2], data[:,3]]).T)
	Pw = W*(W.T*W).I*W.T
	ivcoef = (x.T*Pw*x).I*x.T*Pw*y
	res = y-x*ivcoef
	g_x = np.multiply(W,res)	
	omega = g_x.T*g_x/data.shape[0]
	G = -W.T*x/data.shape[0]
	var = (G.T*G).I*G.T*omega*G*(G.T*G).I/data.shape[0] #eqn 4.11
	return np.hstack((ivcoef,var))


def getIVEstMR(data): # data has to be n by 4 array
	y = np.matrix(np.array(data[:,0])).T
	x = np.matrix(np.array(data[:,1])).T
	W = np.matrix(np.array([data[:,2], data[:,3]]).T)
	Pw = W*(W.T*W).I*W.T
	ivcoef = (x.T*Pw*x).I*x.T*Pw*y
	res = y-x*ivcoef
	g_x = np.multiply(W,res) # eqn 4.13
	g = W.T*res/data.shape[0]  # eqn 4.13
	gmg = g_x - g.T # eqn 4.13
	G_x = -np.multiply(W,x) # eqn 4.13
	G = -W.T*x/data.shape[0] # eqn 4.13
	GmG  = G_x - G.T # eqn 4.13
	GmG_g = GmG*g # eqn 4.13
	A=np.concatenate((gmg,GmG_g),axis=1) # eqn 4.13
	omega = A.T*A/data.shape[0] # eqn 4.13
	B = np.concatenate((G.T, np.eye(1)), axis=1) # eqn 4.12
	V = B*omega*B.T # eqn 4.9/10
	H = G.T*G # eqn 4.9/10
	var = H.I*V*H.I/data.shape[0]
	#print var
	return np.hstack((ivcoef, var))



def getAsyCI(n,cofvar,q):
	cof = cofvar[0,0]
	var = cofvar[0,1]
	sd = np.sqrt(var)
	lower = cof - q*sd
	upper = cof + q*sd
	#print cof
	#print sd
	#print lower
	#print upper

	if ((lower < 0.0) & (0.0 < upper)):
		return 1.0
	else:
		return 0.0

def getBootCoverage(coef,var,q):
	sd = np.sqrt(var)
	lower = coef - q*sd
	upper = coef + q*sd
	if ((lower < 0.0) & (0.0 < upper)):
		return 1.0
	else:
		return 0.0



