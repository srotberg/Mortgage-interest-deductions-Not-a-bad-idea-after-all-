import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.weightstats import DescrStatsW
import scipy.ndimage as nd
import statsmodels.api as sm

#################################################################################
# load the big dataset

# X805(#1)        What is the amount still owed on the land contract?
# X905(#2)        
# X1005(#3)

# X816(#1)        What is the current annual rate of interest being charged
# X916(#2)        on the (loan/land contract)?
# X1016(#3)

# X5744           Did you (or your {husband/wife/partner/spouse} file,
#                or do you expect to file, a Federal Income tax return
#                for 2018?

# X5746           (Did/Will) you and your (husband/wife/partner/spouse) file
#               a joint return, (did/will) you file separately, or (did/will)
#               only one of you file?
#
#                     1.    *Filed jointly
#                     2.    *Filed separately
#                     3.    *Only R Filed
#                     4.    *Only Spouse/Partner Filed
#                     0.     Inap. (did not file and does not expect to do
#                            so: X5744=5; no spouse/partner)

# X7367(#1)       (Did/Will) (you/he/she/he or she) itemize deductions?
# X7368(#2)
# X7369(#3)
#
# X721            What are the real estate taxes on (this home and land/
#                this land/this home/this farm/this ranch/the part of the
#                ranch you own/the part of the farm you own/this property)?
# X722            INTERVIEWER: CODE WITHOUT ASKING IF ALREADY MENTIONED.
#                (And that amount is per...?)
#                
#                FREQUENCY:
#                     2.    *Week
#                     3.    *Every two weeks
#                     4.    *Month
#                     5.    *Quarter
#                     6.    *Year
#                    11.    *Twice per year
#                    12.     Every two months
#                    20.     Five times a year; every 10 weeks
#                    22.     Varies
#                    25.     Every 2 years
#                    31.    *Twice a month
#                    -1.     None
#                    -7.    *Other
#                     0.     Inap. (does not own any part of HU: X508^=(1, 2)
#                            and X601^=(1, 2, 3) and X701^=(1, 3, 4,
#                            5, 6, 8) and X7133^=1)

# X435            Apply for a mortgage or home-based loan?
#
#                     1.    *YES
#                     5.    *NO

# X436            Request to refinance a mortgage?
#
#                     1.    *YES
#                     5.    *NO

# X7137(#1)       Did you take out this (mortgage/loan) to:
#                refinance or rollover an earlier loan, borrow additional
#                money on your home equity, or to do both?
#
#                     1.    *Refinance or rollover an earlier loan,
#                     2.    *Borrow additional money on your home equity,
#                     3.    *Or to do both?
#                     4.    *ORIGINALLY PAID CASH AND TOOK OUT LOAN LATER;
#                            no mortgage when loan taken out; bought land
#                            without a loan, took out construction loan later
#                     8.     Assumed mortgage when inherited the house
#                     0.     Inap. (does not own any part of HU or owns only
#                            mobile home and not site: X508^=(1, 2)
#                            and X601^=(1, 2) and X701^=(1, 3, 4, 5,
#                            6, 8) and X7133^=1; land contact:
#                            X723=2; no mortgage: X723=5; year of
#                            mortgage same as year of purchase:
#                            X802= one of X606, X611, X616,
#                            X630, X634, or X720)



cols = ['yy1','x805','x905','x1005','x816','x916','x1016','x5744',
        'x5746','x7367','x7368','x7369','x721','x722','x703','x704']
#        'x435','x436','x7137']
main = pd.read_stata('~/Documents/Research/Datasets/SCF/p19i6.dta')[cols]

# NOTE: THIS FILE IS TOO LARGE TO UPLOAD TO GITHUB. YOU CAN DOWNLOAD IT FROM 
# THE FED HERE: https://www.federalreserve.gov/econres/files/scf2019s.zip

################################################################################
# load the summary and merge on the required vars from

summary = pd.read_stata('rscfp2019.dta')

df = pd.merge(left=summary,right=main,how='left',on='yy1')

# Filters
df = df[df.income>0]
df = df[df.age>=23]
df = df[df.age<=85]
df=df.reset_index(drop=True)

# Age groups
df.age_grp = 0
df.loc[df.age<35,'age_grp']=0
df.loc[(df.age>=35) & (df.age<45),'age_grp']=1
df.loc[(df.age>=45) & (df.age<55),'age_grp']=2
df.loc[(df.age>=55) & (df.age<65),'age_grp']=3
df.loc[df.age>=65,'age_grp']=4

# compute vars using extra data
df['xx05'] = df.x805+df.x905+df.x1005 # total mortgage balance to check against summary extract
df['i1'] = (df.x816/100/100)*df.x805 # interest in first mortgage
df['i2'] = (df.x916/100/100)*df.x905 # interest on second mortgage
df['i3'] = (df.x1016/100/100)*df.x1005 # interest on third mortgage
df['ii'] = df.i1 + df.i2 + df.i3 # total mortgage interest

df['itemize']=0
df.loc[df.x7367==1,'itemize']=1 # jointly itemized
df.loc[df.x7368==1,'itemize']=1 # reference itemized
df.loc[df.x7368==9,'itemize']=1 # spouse itemized

# compute MID
df['mid']=df.ii
df.loc[df.itemize==0,'mid'] = 0.0

# assign people to income quintiles
wq = DescrStatsW(data=df.income, weights=df.wgt)
probs = np.array([0.2,0.4,0.6,0.8])
quintiles  = wq.quantile(probs=probs, return_pandas=False)
df['grp']=0
for i in range(len(quintiles)):
    q = quintiles[i]
    df.loc[df.income>q,'grp']=i+1

df['protax']=df.x721
df['protax_freq'] = 0
df.loc[df.x722==2,'protax_freq']=52
df.loc[df.x722==3,'protax_freq']=52/2
df.loc[df.x722==4,'protax_freq']=12
df.loc[df.x722==5,'protax_freq']=4
df.loc[df.x722==6,'protax_freq']=1
df.loc[df.x722==11,'protax_freq']=2
df.loc[df.x722==12,'protax_freq']=6
df.loc[df.x722==20,'protax_freq']=5
df.loc[df.x722==25,'protax_freq']=0.5
df.loc[df.x722==31,'protax_freq']=24

df['fee']=df.x703
df['fee_freq'] = 0
df.loc[df.x704==2,'protax_freq']=52
df.loc[df.x704==3,'protax_freq']=52/2
df.loc[df.x704==4,'protax_freq']=12
df.loc[df.x704==5,'protax_freq']=4
df.loc[df.x704==6,'protax_freq']=1
df.loc[df.x704==11,'protax_freq']=2
df.loc[df.x704==12,'protax_freq']=6
df.loc[df.x704==20,'protax_freq']=5
df.loc[df.x704==25,'protax_freq']=0.5
df.loc[df.x704==31,'protax_freq']=24

#################################################################################

def wgt_mean(x):
    tmp = DescrStatsW(data=x, weights=df.loc[x.index,'wgt'])
    return tmp.mean

def wgt_sum(x):
    tmp = DescrStatsW(data=x, weights=df.loc[x.index,'wgt'])
    return tmp.sum

def wgt_med(x):
    wq = DescrStatsW(data=x, weights=df.loc[x.index,'wgt'])
    return wq.quantile(probs=0.5, return_pandas=False)[0]


#################################################################################
# Load the model results

model = pd.read_csv('baseline_benchmark.txt',sep=',',skipinitialspace = True)
model.columns = ['variable','value']
model=model.set_index('variable')
model=model.transpose().reset_index(drop=True)
model = model.loc[:,~model.columns.duplicated()].copy()

nmr = pd.read_csv('nmr_benchmark.txt',sep=',',skipinitialspace = True)
nmr.columns = ['variable','value']
nmr=nmr.set_index('variable')
nmr=nmr.transpose().reset_index(drop=True)
nmr = nmr.loc[:,~nmr.columns.duplicated()].copy()


#################################################################################
# Spending on housing

print('\n-------------------------------------------------------------------------')
print('Spending on housing:\n')

df['rent_to_income'] = df.rent*12 / df.income
df['cost_burdened'] = df.rent_to_income>0.5


df['spending'] = 0
df.loc[df.housecl==2,'spending']=12*df.loc[df.housecl==2,'rent']
df.loc[df.housecl==1,'spending']= (12*df.loc[df.housecl==1,'mortpay'] + 
                                   df.loc[df.housecl==1,'protax']*df.loc[df.housecl==1,'protax_freq'] +
                                   df.loc[df.housecl==1,'fee']*df.loc[df.housecl==1,'fee_freq'])
df['ratio'] = df.spending/df.income


renters = df[df.housecl==2]
renters = renters[(renters['rent_to_income']<1) & (renters['rent_to_income']>0)]


df['ratio'] = df.spending/df.income
mask=df.ratio<1

# store rent to income CDF
ax = sns.ecdfplot(data=renters,x='rent_to_income',weights='wgt')
line = ax.get_lines()[0]
xdata = line.get_data()[0]
ydata = line.get_data()[1]
rent_income_cdf = pd.DataFrame({'rent_to_income':xdata,'data':ydata})
plt.close('all')

print('Data:')
print('Share of cost burdened renters: %0.4f' % (100*renters[renters.rent_to_income>0.5].wgt.sum()/renters.wgt.sum()))
print('Share of renters spending 30+ pct: %0.4f' % (100*renters[renters.rent_to_income>0.30].wgt.sum()/renters.wgt.sum()))
print('Share of renters spending 65+ pct: %0.4f' % (100*renters[renters.rent_to_income>0.65].wgt.sum()/renters.wgt.sum()))
print('Share of renters spending 75+ pct: %0.4f' % (100*renters[renters.rent_to_income>0.75].wgt.sum()/renters.wgt.sum()))

tmp = DescrStatsW(data=df[mask].ratio, weights=df[mask].wgt)
print('Avg spending/income (all): %0.4f' % (100*tmp.mean))


tmp = DescrStatsW(data=renters.rent_to_income, weights=renters.wgt)
print('Avg spending/income (renters): %0.4f' % (100*tmp.mean))

print('\nModel:')
print('Share of cost burdened renters: %0.4f' % ((100*(1-model.rent_to_income_cdf_50[0]))))
print('Share of renters spending 65+ pct: %0.4f' % ((100*(1-model.rent_to_income_cdf_65[0]))))
print('Share of renters spending 75+ pct: %0.4f' % ((100*(1-model.rent_to_income_cdf_75[0]))))
print('Avg spending/income (all): %0.4f' % model.avg_housing_spending_all[0])
print('Avg spending/income (renters): %0.4f' % model.avg_housing_spending_renters[0])

# life cycle of cost-burden
life_cycle = renters.groupby('age').cost_burdened.aggregate(lambda x: wgt_mean(x)).reset_index()

#################################################################################
# Homeownership rate

print('\n-------------------------------------------------------------------------')
print('Homeownership:\n')


df['ho'] = df.housecl==1
ho_by_inc = df.groupby('grp')['ho'].aggregate(lambda x: wgt_mean(x)).reset_index()
tmp = df.groupby('age').ho.aggregate(lambda x: wgt_mean(x)).reset_index()
life_cycle = pd.merge(left=life_cycle,right=tmp,how='left',on='age')
life_cycle['cost_burdened_alt'] = life_cycle.cost_burdened*(1.0-life_cycle.ho)

print('Data:')
print('Aggregate HO rate: %0.4f' % (100*df[df.ho==1].wgt.sum()/df.wgt.sum()))
print('Aggregate HO rate (old): %0.4f' % (100*df[(df.ho==1)&(df.age>65)].wgt.sum()/df[df.age>65].wgt.sum()))
labels=['1st','2nd','3rd','4th','5th']
print('Pct. renters by income quintile:')
for i in range(len(ho_by_inc)):
    print('\t' +labels[i] + '\t%0.4f' % (100*(1-ho_by_inc.ho[i])))

print('\nModel:')
print('Aggregate HO rate: %0.4f' % (model.HO[0]))
print('Pct. renters by income quintile:')
for i in range(len(ho_by_inc)):
    print('\t' +labels[i] + '\t%0.4f' % (model['renters_income_quintile_%d'%(i+1)][0]))

#################################################################################
# Mortgages

print('\n-------------------------------------------------------------------------')
print('Mortgages:\n')

df['mortgage'] = df['mrthel']>0

df['ltv']=0
df.loc[df.housecl==1,'ltv'] = df.loc[df.housecl==1,'mrthel']/df.loc[df.housecl==1,'houses']

df['house_to_income'] = 0
df.loc[df.housecl==1,'house_to_income'] = df.loc[df.housecl==1,'houses']/df.loc[df.housecl==1,'income']

#df['refinance_2019'] = df['x436']==1
#df['refinance_ever'] = (df['x7137']==1) & (df['ltv']>0.0001)

homeowners = df.loc[df.housecl==1]
mortgagers = df.loc[(df.housecl==1) & (df.ltv>0.001)]

tmp = homeowners.groupby('age')['ltv'].agg(lambda x: wgt_med(x)).reset_index()
life_cycle = pd.merge(left=life_cycle,right=tmp,how='left',on='age')

#tmp = mortgagers.groupby('age')['refinance_ever'].agg(lambda x: wgt_mean(x)).reset_index()
#life_cycle = pd.merge(left=life_cycle,right=tmp,how='left',on='age')


wq = DescrStatsW(data=homeowners.mortgage, weights=homeowners.wgt)

print('Data:')
print('Mean(house value/income) (all): %0.4f' % (wgt_mean(df.loc[(df.ho==1),'house_to_income'])))
print('Mean(house value)/mean(income): %0.4f' % (wgt_mean(df.loc[(df.ho==1)&(df.houses>10000),'houses'])/wgt_mean(df.income)))
#print('Mean(house value/income) (working age): %0.4f' % (wgt_mean(df.loc[(df.ho==1) & (df.age<66),'house_to_income'])))
#print('House value/income: %0.4f' % (wgt_sum(df.houses)/wgt_sum(df.income)))
print('Mortgage debt/house value: %0.4f' % (wgt_sum(df.mrthel)/wgt_sum(df.houses)))
print('Pct. homeowners with mortgage: %0.4f' % (100*wq.mean))
print('Pct.  homeowners with LTV>20: %0.4f' % (100*homeowners.wgt[homeowners.ltv>0.2].sum()/homeowners.wgt.sum()))
print('Pct.  homeowners with LTV>35: %0.4f' % (100*homeowners.wgt[homeowners.ltv>0.315].sum()/homeowners.wgt.sum()))
print('Pct.  homeowners with LTV>80: %0.4f' % (100*homeowners.wgt[homeowners.ltv>0.8].sum()/homeowners.wgt.sum()))
print('Pct.  homeowners with LTV>90: %0.4f' % (100*homeowners.wgt[homeowners.ltv>0.9].sum()/homeowners.wgt.sum()))
#print('Pct. mortgage holders refinance this year: %0.4f' % (100* mortgagers.wgt[mortgagers.refinance_2019].sum()/mortgagers.wgt.sum()))
#print('# refinanced mortgages/all mortgages: %0.4f' % (100*mortgagers.wgt[mortgagers.refinance_ever].sum()/mortgagers.wgt.sum()))
#print('$ refinanced mortgages/all mortgages: %0.4f' % (100*wgt_sum(df.mrthel[df.refinance_ever])/wgt_sum(df.mrthel)))

print('\nModel:')
print('Mean(house value/income): %0.4f' % (model['avg_(owned_house_to_income)'][0]))
print('Mean(house value)/mean(income): %0.4f' % (model['avg_owned_house_to_avg_owner_income'][0]))
print('Mortgage debt/house value: %0.4f' %  model.mortgage_debt_to_housing_wealth[0])
print('Pct. homeowners with mortgage: %0.4f' % (model.HO_with_morgage[0]))
print('Pct.  homeowners with LTV>80: %0.4f' % (model.HO_with_mortgage_over_80_perc[0]))
print('Pct.  homeowners with LTV>90: %0.4f' % (model.HO_with_mortgage_over_90_perc[0]))

# store LTV CDF
ax = sns.ecdfplot(data=homeowners[homeowners.ltv<=1],x='ltv',weights='wgt')
line = ax.get_lines()[0]
xdata = line.get_data()[0]
ydata = line.get_data()[1]
ltv_cdf = pd.DataFrame({'ltv':xdata,'data':ydata})
plt.close('all')
#ltv_cdf = ltv_cdf[


#################################################################################
# Savings + net worth

print('\n-------------------------------------------------------------------------')
print('Savings + net worth by age group:\n')

df['fin_income'] = df.fin/df.income
df['nw_income'] = df.networth/df.income

tmp = df.groupby(['age','age_grp']).networth.aggregate(lambda x: wgt_med(x)).reset_index()
tmp2 = tmp.groupby('age_grp').networth.mean().reset_index()
tmp2.networth=tmp2.networth/tmp2.networth[4]
tmp=tmp.drop('age_grp',axis=1)

life_cycle = pd.merge(left=life_cycle,right=tmp,how='left',on='age')

#labels=['<35','35-44','45-54','55-64','>65']
#print('Med nw (rel. age 65+):')
#for i in range(len(tmp2)):
#    print('\t' +labels[i] + '\t%0.4f' % (tmp2.networth[i]/tmp2.networth[4]))

print('Data:')
print('Med NW age 55-64 / Med NW age 65+ = %0.4f' % (tmp2.networth[3]/tmp2.networth[4]))

tmp3 = 0
cnt=0
for i in range(55,65):
    tmp3 = tmp3 + model['median_assets_age_%d'%i][0]
    cnt = cnt+1
tmp3 = tmp3/cnt

tmp4 = 0
cnt=0
for i in range(65,86):
    tmp4 = tmp4 + model['median_assets_age_%d'%i][0]
    cnt = cnt+1
tmp4 = tmp4/cnt

print('\nModel:')
print('Med NW age 55-64 / Med NW age 65+ = %0.4f' % (tmp3/tmp4))
    
    
################################################################################
# Net worth of the young

print('\n-------------------------------------------------------------------------')
print('Wealth distribution of ages 23-27:')

def frac_above_threshold(x,threshold=1000):
    weights = df.loc[x.index,'wgt']
    return weights[x>threshold].sum()/weights.sum()
    
def med_above_threshold(x,threshold=1000,return_pandas=False):
    x2=x[x>threshold]
    wq = DescrStatsW(data=x2, weights=df.loc[x2.index,'wgt'])
    return wq.quantile(probs=0.5, return_pandas=return_pandas)[0]

wa = df[(df.age>=25) & (df.age<65)]
wq = DescrStatsW(data=wa.income, weights=wa.wgt)
medinc = wq.quantile(probs=0.5, return_pandas=False)[0]

young = df.loc[(df.age>=23) & (df.age<=27),:]
wq = DescrStatsW(data=young.income, weights=young.wgt)
probs = np.array([0.05,0.1,0.4,0.7,0.9])
labs = ['z=1,2','z=3','z=4','z=5','z=6','z=7-9']
pct  = wq.quantile(probs=probs, return_pandas=False)

young.loc[young.index,'grp']=0
for i in range(len(pct)):
    p = pct[i]
    young.loc[young.income>p,'grp']=i+1

young_initial_wealth = young.groupby('grp').aggregate(frac_pos=pd.NamedAgg(column='networth',aggfunc = lambda x: frac_above_threshold(x,1000)),
                                                      med_pos=pd.NamedAgg(column='networth',aggfunc = lambda x: med_above_threshold(x,1000)))
young_initial_wealth['normalized'] = young_initial_wealth.med_pos/medinc

print('Model only:')
print('\tZ\tFrac w/ NW\tMedian/Avg inc')
for i in range(len(young_initial_wealth)):
    print('\t' + labs[i] + '\t%0.4f\t\t%0.4f'%(young_initial_wealth.frac_pos[i],young_initial_wealth.normalized[i]))

############################################################################
# load the model data

life_cycle = life_cycle[(life_cycle.age>=26) & (life_cycle.age<86)].reset_index(drop=True)

cols = ['ho','ltv','networth','cost_burdened','cost_burdened_alt']
for c in cols:
    life_cycle[c+'2'] = sm.nonparametric.lowess(exog=life_cycle.age,
                                                endog=life_cycle[c],
                                                frac=0.3,
                                                return_sorted=False)

ratios = np.array(range(0,101))
ltvs = np.array(range(0,101))
ages = np.array(range(26,86))

model_cdf_baseline = np.zeros(len(ratios))
model_cdf_ltv = np.zeros(len(ratios))
for i in range(len(ratios)):
    model_cdf_baseline[i] = model['rent_to_income_cdf_%d'%(ratios[i])][0]
    model_cdf_ltv[i] = model['LTV_cdf_%d'%(ltvs[i])][0]

model_cdf_nmr = np.zeros(len(ratios))
for i in range(len(ratios)):
    model_cdf_nmr[i] = nmr['rent_to_income_cdf_%d'%(ratios[i])][0]    


model_lf_ho = np.zeros(len(ages))
model_lf_ltv = np.zeros(len(ages))
model_lf_nw = np.zeros(len(ages))
model_lf_cb = np.zeros(len(ages))
model_lf_cb2 = np.zeros(len(ages))
for i in range(len(ages)):
    a = ages[i]
    model_lf_ho[i] = model['HO_age_%d'%a][0]
    model_lf_ltv[i] = model['LTV_age_%d'%a][0]
    model_lf_nw[i] = model['median_assets_age_%d'%a][0]
    model_lf_cb[i] = (model['cost_burdened_renters_age_%d'%a][0])
    model_lf_cb2[i] = (model['cost_burdened_renters_age_%d'%a][0])*(1.0-model_lf_ho[i]/100)


############################################################################
# make plots

mpl.rc('savefig',format='pdf')
mpl.rcParams['lines.markersize'] = 2.5
mpl.rcParams['savefig.pad_inches'] = 0

lw=3
tw=20
alpha=0.8
colors=['#377eb8','#e41a1c','#4daf4a','#984ea3', '#ff7f00','#ffff33']
alpha=0.8
dashes=[(None,None),(12,6),(6,3),(3,1.5)]
markers=[None,'o','s','^']

def paper_fig(sz=(3.85,3)):
    fig, ax = plt.subplots(figsize=sz,tight_layout = {'pad': 0})
    ax.tick_params(axis='both', labelsize=10)
    ax.yaxis.label.set_size(10)
    #sns.despine()   
    return fig,ax

# rent to income CDF
#fig,ax = paper_fig((4.25,3))
fig,ax = paper_fig()
ax.plot(rent_income_cdf.rent_to_income,rent_income_cdf.data,color=colors[0],alpha=0.8,label='Data (SCF)',linestyle='-')
#ax.axvline(0.5,color='black',lw=0.5)
ax.set_xlabel('Rent-income ratio')
ax.set_ylabel('Fraction of renters')
plt.savefig('cdf0.pdf',bbox_inches='tight')
ax.plot(ratios/100,model_cdf_baseline,color=colors[1],alpha=0.8,label='Benchmark calibration',linestyle='--')
ax.plot(ratios/100,model_cdf_nmr,color=colors[2],alpha=0.8,label='No min. rental',linestyle=':')
ax.legend(loc='lower right',prop={'size':7})
plt.savefig('cdf.pdf',bbox_inches='tight')
plt.close('all')

# LTV CDF
fig,ax = paper_fig()
ax.plot(ltv_cdf.ltv,ltv_cdf.data,color=colors[0],alpha=0.8,label='Data (SCF)',linestyle='-')
ax.plot(ltvs/100,model_cdf_ltv,color=colors[1],alpha=0.8,label='Model',linestyle='--')
#ax.axvline(0.5,color='black',lw=0.5)
ax.set_xlabel('Loan-to-value ratio')
ax.set_ylabel('Fraction of homeowners')
ax.set_xlim(1e-6,1)
ax.legend(loc='lower right',prop={'size':7})
plt.savefig('cdf_ltv.pdf',bbox_inches='tight')
plt.close('all')

# home ownership by age grp
fig,ax = paper_fig()
ax.plot(life_cycle.age,100*life_cycle.ho,color=colors[0],alpha=0.5,label='Data',lw=0.75,ls=':')
ax.plot(life_cycle.age,100*life_cycle.ho2,color=colors[0],alpha=0.8,label='Data (smoothed)')
ax.plot(ages,model_lf_ho,color=colors[1],alpha=0.8,label='Model',linestyle='--')
ax.legend(loc='lower right',prop={'size':7})
ax.set_xlim(26,85)
ax.set_xticks([30,40,50,60,70,80])
plt.savefig('ho_by_age.pdf',bbox_inches='tight')
#ax.plot(model_lf.age,100*model_lf['ho MID repeal'],color=colors[2],alpha=0.8,label='Model after MID repeal',linestyle=':')
#ax.legend(loc='lower right',prop={'size':7})
#plt.savefig('ho_by_age_wo_mid.pdf',bbox_inches='tight')
plt.close('all')

# ltv by age grp
fig,ax = paper_fig()
ax.plot(life_cycle.age,100*life_cycle.ltv,color=colors[0],alpha=0.5,label='Data',lw=0.75,ls=':')
ax.plot(life_cycle.age,100*life_cycle.ltv2,color=colors[0],alpha=0.8,label='Data (smoothed)')
ax.plot(ages,model_lf_ltv,color=colors[1],alpha=0.8,label='Model',linestyle='--')
ax.set_xlim(26,85)
ax.set_xticks([30,40,50,60,70,80])
ax.legend(loc='upper right',prop={'size':7})
plt.savefig('ltv_by_age.pdf',bbox_inches='tight')
plt.close('all')

# nw by age
fig,ax = paper_fig()
ax.plot(life_cycle.age,life_cycle.networth/life_cycle.networth[40:].mean(),color=colors[0],alpha=0.75,label='Data',lw=0.5,ls=':')
ax.plot(life_cycle.age,life_cycle.networth2/life_cycle.networth2[40:].mean(),color=colors[0],alpha=0.8,label='Data (smoothed)')
ax.plot(ages,model_lf_nw/model_lf_nw[40:].mean(),color=colors[1],alpha=0.8,label='Model',linestyle='--')
ax.set_xlim(26,85)
ax.set_xticks([30,40,50,60,70,80])
ax.legend(loc='upper left',prop={'size':7})
plt.savefig('nw_by_age.pdf',bbox_inches='tight')
plt.close('all')

# cost burdened renters by age
fig,ax = paper_fig()
ax.plot(life_cycle.age,life_cycle.cost_burdened,color=colors[0],alpha=0.5,label='Data',ls=':',lw=0.75)
ax.plot(life_cycle.age,life_cycle.cost_burdened2,color=colors[0],alpha=0.8,label='Data (smoothed)')
ax.plot(ages,model_lf_cb,color=colors[1],alpha=0.8,label='Model',linestyle='--')
ax.set_ylim(0,0.5)
ax.set_xlim(26,85)
ax.set_xticks([30,40,50,60,70,80])
ax.legend(loc='upper left',prop={'size':7})
plt.savefig('cb_by_age.pdf',bbox_inches='tight')
plt.close('all')

# cost burdened households by age
fig,ax = paper_fig()
ax.plot(life_cycle.age,life_cycle.cost_burdened_alt,color=colors[0],alpha=0.5,label='Data',ls=':',lw=0.75)
ax.plot(life_cycle.age,life_cycle.cost_burdened_alt2,color=colors[0],alpha=0.8,label='Data (smoothed)')
ax.plot(ages,model_lf_cb2,color=colors[1],alpha=0.8,label='Model',linestyle='--')
ax.set_ylim(0,0.15)
ax.set_xlim(26,85)
ax.set_xticks([30,40,50,60,70,80])
ax.legend(loc='upper left',prop={'size':7})
plt.savefig('cb2_by_age.pdf',bbox_inches='tight')
plt.close('all')


#################################################################################
# MID calculations

print('\n-------------------------------------------------------------------------')
print('MID incidence:\n')

res1 = df.groupby('grp')['itemize'].aggregate(lambda x: wgt_mean(x))

labels=['1st','2nd','3rd','4th','5th']
print('Share of itemizers by income quintile:')
for i in range(len(res1)):
    print('\t' + labels[i] + '\t%0.4f' % res1[i])

# share of MID by quintile
wq = DescrStatsW(data=df.mid, weights=df.wgt)
total_sum = wq.sum

def wgt_share(x):
    tmp = DescrStatsW(data=x, weights=df.loc[x.index,'wgt'])
    return tmp.sum/total_sum

res2 = df.groupby('grp')['mid'].aggregate(lambda x: wgt_share(x))

print('\nShare of MID by income quintile:')
for i in range(len(res2)):
    print('\t' + labels[i] + '\t%0.4f' % res2[i])
