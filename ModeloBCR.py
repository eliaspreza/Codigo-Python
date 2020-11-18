#!/usr/bin/env python
# coding: utf-8

# # ANEXO: Modelo para estimar sensibilidad riesgo sistémico

# In[1]:


#============================================================================
#==============Modelo de regresion para estimar riesgo sistemico=============
#=============================================================================

#---cargando las librerias

from statsmodels.compat import lzip

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import statsmodels.stats.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.sandbox.regression.predstd import wls_prediction_std




# In[2]:


#---Cargando la base de datos

base=pd.read_excel('F:\CONSULTORIAS\Financiero\Para Prueba\ModeloBCR_Py\Base.xlsx', 'Hoja1', index_col=None, na_values=['NA'])
base.head(3)


# In[3]:


#----Comprobando las correlaciones
d=base.corr()
d.head(3)


# In[4]:


#--Graficando las correlaciones
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(d, annot=True,linewidths=.5, ax=ax)


# In[5]:


#--guardando las correlaciones
d.to_excel('F:\CONSULTORIAS\Financiero\Para Prueba\ModeloBCR_Py\Correlaciones.xlsx', sheet_name='Corr')


# In[6]:


#--Creando el modelo
results2=smf.ols('LiquidezTotal ~ r_liquidez+r_patrimonio+np.log(r_morosidad1)+base.r_riesgo1', data=base).fit()
print(results2.summary())


# In[7]:


#--Normalidad
#--Prueba Jarque Bera
name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
test = sms.jarque_bera(results2.resid)
lzip(name, test)


# In[8]:


#--Normalidad
#--Omni test:
name = ['Chi^2', 'Two-tail probability']
test = sms.omni_normtest(results2.resid)
lzip(name, test)


# In[9]:


#--Influence tests
from statsmodels.stats.outliers_influence import OLSInfluence
test_class = OLSInfluence(results2)
test_class.dfbetas[:5,:]


# In[10]:


from statsmodels.graphics.regressionplots import plot_leverage_resid2
fig = plot_leverage_resid2(results2, ax = ax)


# In[11]:


#--Multicolinealidad
np.linalg.cond(results2.model.exog)


# In[12]:


#--Heteroskedasticity tests (Breush-Pagan test)
name = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(results2.resid, results2.model.exog)
lzip(name, test)


# In[13]:


#--Goldfeld-Quandt test
name = ['F statistic', 'p-value']
test = sms.het_goldfeldquandt(results2.resid, results2.model.exog)
lzip(name, test)


# In[14]:


resid_val=results2.resid
fitted_val=results2.predict()
results2.resid.mean()


# In[15]:


#------Modelo 2
x=pd.DataFrame({'Liquidez':base.r_liquidez,'Patrimonio':base.r_patrimonio,'Morosidad':np.log(base.r_morosidad1),'Riesgo':base.r_riesgo1})
x.head(5)


# In[16]:


y=base.LiquidezTotal


# In[17]:


y


# In[18]:


import statsmodels.api as sm
x_constant=sm.add_constant(x)
Modelo=sm.OLS(y,x_constant).fit()
Modelo.summary()


# In[19]:


residuos=Modelo.resid
fitted=Modelo.predict()
Modelo.resid.mean()


# In[20]:


#--Normalidad
from scipy import stats
stats.shapiro(Modelo.resid)
sm.qqplot(residuos,line='s')


# In[21]:


stats.shapiro(Modelo.resid)


# In[22]:


#--Linealidad
sns.regplot(x=fitted,y=y, lowess=True,line_kws={'color':'red'})
plt.title('Fitted vrs Observados')


# In[23]:


#--Homocedasticidad
resid_stand=Modelo.get_influence().resid_studentized_internal
sns.regplot(x=fitted,y=resid_stand,lowess=True,line_kws={'color':'red'})
plt.title('Fitted vrs Residuals')


# In[24]:


#--Influencia
fig,ax=plt.subplots(figsize=(12,8))
sm.graphics.influence_plot(Modelo,alpha=0.05,ax=ax,criterion='cooks')


# In[25]:


#---Homosedasticidad
bp_test=sms.het_breuschpagan(residuos,Modelo.model.exog)
print(bp_test)
print("Breuschpagan test: pvalue =",bp_test[1])


# In[26]:


#---Autocorrelacion
from scipy.stats.stats import pearsonr
x.columns
pearsonr(x['Liquidez'],Modelo.resid)


# In[27]:


sns.heatmap(x.corr(),annot=True)


# In[28]:


#---Multicolinealidad
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif=[variance_inflation_factor(x_constant.values,i) for i in range (x_constant.shape[1])]
pd.DataFrame({'vif':vif[1:]}, index=x.columns).T


# In[29]:


fig, ax = plt.subplots(figsize=(12, 8))
fig = sm.graphics.plot_fit(Modelo, "Liquidez", ax=ax)


# In[30]:


fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(Modelo, fig=fig)


# In[56]:


print('Parameters: ', Modelo.params)
print('Standard errors: ', Modelo.bse)
print('Predicted values: ', Modelo.predict())


# In[42]:


Modelo.fittedvalues


# In[43]:


y


# In[46]:


x=Modelo.fittedvalues
x


# In[47]:


tabla=pd.DataFrame({'x':x,'y':y })
tabla


# In[53]:


fig,ax=plt.subplots(figsize=(9, 6))
plt.plot(tabla.x,'-g',label="Datos",marker='x')
plt.plot(tabla.y,'--r',label="Predicción")
plt.title("Modelo vrs Datos", size=18)
plt.xlabel("Periodo")
plt.ylabel("Millones de US$")
plt.legend(loc="upper left")


# In[50]:


fig, ax = plt.subplots()
ax.scatter(x, y, alpha=0.5)

ax.set_xlabel(r'$\Delta_i$', fontsize=15)
ax.set_ylabel(r'$\Delta_{i+1}$', fontsize=15)
ax.set_title('Volume and percent change')

ax.grid(True)
fig.tight_layout()

plt.show()


# In[5]:


help(scatter)

