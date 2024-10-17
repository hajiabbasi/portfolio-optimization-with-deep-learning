import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats 


def drawdown(return_series: pd.Series):
    """یک سری زمانی بازده را دریافت میکند
       و سه ستون شاخص, پیک شاخصی و حداکثر سقوط را خروجی میدهد
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})



def get_ffme_returns():

    rets = pd.read_csv("Portfolios_Formed_on_monthly.csv", index_col=0, parse_dates=True, na_values=-99.99)
    rets = rets[["Lo 10", "Hi 10"]] / 100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period("M")
    return rets 



def get_hfi_returns():
    '''
    نرخ بازده هج فاند را بر می گرداند
    '''

    hfi = pd.read_csv("hedgefundindices.csv", index_col=0, parse_dates=True, na_values=-99.99) / 100.0
    return hfi 

def get_ind_returns():

    ind = pd.read_csv("ind30_m_vw_rets.csv", header=0, index_col=0)/100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind


def skewness(s):
    '''
    scipy.stats.skew() ===> همچنین از این تابع می توان استفاده کرد
    '''
    return ( ((s - s.mean()) / s.std(ddof=0))**3 ).mean()

def kurtosis(s):
    '''
    scipy.stats.kurtosis() ===> همچنین از این تابع می توان استفاده کرد
   "Excess Kurtosis" ====> کشیدگی منهای 3
    '''
    return ( ((s - s.mean()) / s.std(ddof=0))**4 ).mean()


def is_normal(s, level=0.01):
    '''
    آزمایش چارک-برا برای دیدن اینکه آیا یک سری بازده دارای توزیع نرمال است یا خیر.
    تست به طور پیش فرض در سطح 1% اعمال می شود
    '''
    statistic, pvalue = scipy.stats.jarque_bera( s )
    return pvalue > level


def semideviation(r):
    """
    نیمه انحراف را برمی گرداند
    r ===>  باید یک سری یا یک دیتافریم باشد، در غیر این صورت یک ارور ایجاد می کند
    """
    if isinstance(r, pd.Series):
        is_negative = r < 0
        return r[is_negative].std(ddof=0)
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(semideviation)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
        
        
        
def var_historic(r, level=5):
    """
    ارزش در معرض خطر تاریخی را در سطح مشخصی برمی‌گرداند
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame") 
        
        
        
from scipy.stats import norm
def var_gaussian(r, level=5, modified=False):
    """
    If "modified" is True ===> ارزش در معرض تعدیل شده را بر میگرداند
    using the Cornish-Fisher modification
    """
    # compute the Z score
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))


def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)    # علامت منفی برای اینه که خود وار ها منفی هستند
        return -r[is_beyond].mean()
    
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


        
def annualize_rets(s, periods_per_year):
    '''
    بازده سالانه را محاسبه میکند
    periods_per_year ===> 12, 52, 252, 
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( annualize_rets, periods_per_year = periods_per_year )
    elif isinstance(s, pd.Series):
        growth = (1 + s).prod()
        n_period_growth = s.shape[0]
        return growth**(periods_per_year/n_period_growth) - 1


    
def annualize_vol(s, periods_per_year, ddof=1):
    '''
    نوسان سالانه را محاسبه میکند
    periods_per_year ===> 12, 52, 252,  
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate(annualize_vol, periods_per_year = periods_per_year )
    elif isinstance(s, pd.Series):
        return s.std(ddof=ddof) * (periods_per_year)**(0.5)
    elif isinstance(s, list):
        return np.std(s, ddof=ddof) * (periods_per_year)**(0.5)
    elif isinstance(s, (int,float)):
        return s * (periods_per_year)**(0.5)
    
    
def sharpe_ratio(s, risk_free_rate, periods_per_year, v=None):
    '''
    محاسبه نسبت شارپ سالانه 
    periods_per_year ===> 12, 52, 252,  
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( sharpe_ratio, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year, v=None)
    
    elif isinstance(s, pd.Series):
        # نرخ سالانه بدون ریسک را به دوره تبدیل میکنیم با این فرض که:
        # RFR_year = (1+RFR_period)^{periods_per_year} - 1.
        rf_to_period = (1 + risk_free_rate)**(1/periods_per_year) - 1        
        excess_return = s - rf_to_period
        # اکنون، بازده اضافی را سالانه می کنیم
        ann_ex_rets = annualize_rets(excess_return, periods_per_year)
        # محاسبه نوسانات سالانه
        ann_vol = annualize_vol(s, periods_per_year)
        return ann_ex_rets / ann_vol
    
    elif isinstance(s, (int,float)) and v is not None:
        
        return (s - risk_free_rate) / v
    
    
def portfolio_return(weights, returns):
    """
    بازده پورتفولیو را از بازده و وزن تشکیل دهنده محاسبه می کند
    weights ====> numpy array or Nx1 matrix  
    returns ====> numpy array or Nx1 matrix
    """
    return weights.T @ returns
    
       

def portfolio_vol(weights, cov_rets):


    return ( np.dot(weights.T, np.dot(cov_rets, weights)) )**(0.5) 


def plot_ef2(n_points, er, cov):
    """
    مرز کارا 2 دارایی را ترسیم می کند
    """
    if er.shape[0] != 2 :
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style=".-")






from scipy.optimize import minimize
def minimize_volatility(rets, covmatrix, target_return=None):

    n_assets = rets.shape[0]    
    # حدس اولیه
    init_guess = np.repeat(1/n_assets, n_assets)
    weights_constraint = {
        "type": "eq",
        "fun": lambda w: 1.0 - np.sum(w)  
    }
    if target_return is not None:
        return_constraint = {
            "type": "eq",
            "args": (rets,),
            "fun": lambda w, r: target_return - portfolio_return(w, r)
        }
        constr = (return_constraint, weights_constraint)
    else:
        constr = weights_constraint
        
    result = minimize(portfolio_vol, 
                      init_guess,
                      args = (covmatrix,),
                      method = "SLSQP",
                      options = {"disp": False},
                      constraints = constr,
                      bounds = ((0.0,1.0),)*n_assets ) 
    return result.x




def optimal_weights(n_points, rets, covmatrix, periods_per_year):

    target_rets = np.linspace(rets.min(), rets.max(), n_points)    
    weights = [minimize_volatility(rets, covmatrix, target) for target in target_rets]
    return weights


def plot_ef(n_points, er, cov, periods_per_year):
    """
    مرز کارا چند دارایی را ترسیم می کند
    """
    weights = optimal_weights(n_points, er, cov,periods_per_year)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style='.-')


def compute_returns(s):

    if isinstance(s, pd.DataFrame):
        return s.aggregate( compute_returns )
    elif isinstance(s, pd.Series):
        return s / s.shift(1) - 1
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")
        
        
        
def maximize_sharpe_ratio(rets, covmatrix, risk_free_rate, periods_per_year, target_volatility=None):
    '''
    وزن های بهینه بالاترین نسبت شارپ را در مرز کارا را برمی گرداند. 
    if target_volatility is not None ===>  وزن ها با بالاترین نسبت شارپ پرتفوی را نشان میدهد .
    این روش از بهینه ساز مینیماز  استفاده می کند که مشکل حداکثر کردن نسبت شارپ را حل می کند که معادل به حداقل رساندن نسبت شارپ منفی است.
    '''
    n_assets   = rets.shape[0] 
    init_guess = np.repeat(1/n_assets, n_assets)
    weights_constraint = {
        "type": "eq",
        "fun": lambda w: 1.0 - np.sum(w)  
    }
    if target_volatility is not None:
        volatility_constraint = {
            "type": "eq",
            "args": (covmatrix, periods_per_year),
            "fun": lambda w, cov, p: target_volatility - annualize_vol(portfolio_vol(w, cov), p)
        }
        constr = (volatility_constraint, weights_constraint)
    else:
        constr = weights_constraint
        
    def neg_portfolio_sharpe_ratio(weights, rets, covmatrix, risk_free_rate, periods_per_year):
        '''
        نسبت شارپ سالانه منفی را برای مشکل کمینه‌سازی پورتفولیوهای بهینه محاسبه می‌کند.
        The variable periods_per_year ===> 12, 52, 252.
    
        '''
        # بازده سالانه پرتفوی
        portfolio_ret = portfolio_return(weights, rets)        
        
        # نوسانات سالانه پرتفوی
        portfolio_vola = annualize_vol(portfolio_vol(weights, covmatrix), periods_per_year)
        return -sharpe_ratio(portfolio_ret, risk_free_rate, periods_per_year, v=portfolio_vola)    
        #-(portfolio_ret - risk_free_rate)/portfolio_vol
        
    result = minimize(neg_portfolio_sharpe_ratio,
                      init_guess,
                      args = (rets, covmatrix, risk_free_rate, periods_per_year),
                      method = "SLSQP",
                      options = {"disp": False},
                      constraints = constr,
                      bounds = ((0.0,1.0),)*n_assets)
    return result.x


def efficient_frontier(n_portfolios, rets, covmat, periods_per_year, risk_free_rate=0.0, 
                       iplot=False, hsr=False, cml=False, mvp=False, ewp=False):
    ''' 
    if iplot=True ===> این روش یک دیتافریم حاوی نوسانات، بازده ها، نسبت های شارپ و وزن پورتفولیوها و همچنین نموداری از مرز کارا را برمی گرداند
     
    دیگر ورودی ها:
        hsr ===> if true ===> سبد با بالاترین بازده را رسم میکند
        cml ===> if true ===> خط بازار سرمایه را رسم میکند 
        mvp ===> if true ===> پرتفولیو با حداقل ریسک را رسم میکند
        ewp ===> if true ===> پرتفوی با وزن های برابر را رسم میکند 
        periods_per_year ===> 12, 52, 252, monthly, weekly, daily.
    '''   
    
    def append_row_df(df,vol,ret,spr,weights):
        temp_df = list(df.values)
        temp_df.append( [vol, ret, spr,] + [w for w in weights] )
        return pd.DataFrame(temp_df)
        
    ann_rets = annualize_rets(rets, periods_per_year)
    
    # وزن های بهینه پورتفولیوهایی را که در مرزهای کارا قرار دارند تولید می کند
    weights = optimal_weights(n_portfolios, ann_rets, covmat, periods_per_year) 
    # همچنین اگر سبد فقط از دو دارایی تشکیل شده باشد، وزن ها می توانند به صورت زیر باشند: 
    #weights = [np.array([w,1-w]) for w in np.linspace(0,1,n_portfolios)]

    # portfolio returns
    portfolio_ret = [portfolio_return(w, ann_rets) for w in weights]
    
    # portfolio volatility
    vols          = [portfolio_vol(w, covmat) for w in weights] 
    portfolio_vola = [annualize_vol(v, periods_per_year) for v in vols]
    
    # portfolio sharpe ratio
    portfolio_spr = [sharpe_ratio(r, risk_free_rate, periods_per_year, v=v) for r,v in zip(portfolio_ret,portfolio_vola)]
    
    df = pd.DataFrame({"volatility": portfolio_vola,
                       "return": portfolio_ret,
                       "sharpe ratio": portfolio_spr})
    df = pd.concat([df, pd.DataFrame(weights)],axis=1)
    
    if iplot:
        ax = df.plot.line(x="volatility", y="return", style="--", color="coral", grid=True, label="Efficient frontier", figsize=(8,4))
        if hsr or cml:
            w   = maximize_sharpe_ratio(ann_rets, covmat, risk_free_rate, periods_per_year)
            ret = portfolio_return(w, ann_rets)
            vol = annualize_vol( portfolio_vol(w,covmat), periods_per_year)
            spr = sharpe_ratio(ret, risk_free_rate, periods_per_year, v=vol)
            df  = append_row_df(df,vol,ret,spr,w)
            if cml:
                # Draw the CML====> نقطه های خط بازار سرمایه [0,risk_free_rate] و [port_vol,port_ret]
                ax.plot([0, vol], [risk_free_rate, ret], color="g", linestyle="-.", label="CML")
                ax.set_xlim(left=0)
                ax.legend()
            if hsr:
                # بالاترین نسبت شارپ را رسم میکند
                ax.scatter([vol], [ret], marker="o", color="g", label="MSR portfolio")
                ax.legend()
        if mvp:
            # پرتفو با حداقل نوسانات و ریسک را رسم میکند
            w   = minimize_volatility(ann_rets, covmat)
            ret = portfolio_return(w, ann_rets)
            vol = annualize_vol( portfolio_vol(w,covmat), periods_per_year)
            spr = sharpe_ratio(ret, risk_free_rate, periods_per_year, v=vol)
            df  = append_row_df(df,vol,ret,spr,w)
            ax.scatter([vol], [ret], color="midnightblue", marker="o", label="GMV portfolio")
            ax.legend()  
        if ewp:
            # پرتفو با وزن های برابر را رسم میکند
            w   = np.repeat(1/ann_rets.shape[0], ann_rets.shape[0])
            ret = portfolio_return(w, ann_rets)
            vol = annualize_vol( portfolio_vol(w,covmat), periods_per_year)
            spr = sharpe_ratio(ret, risk_free_rate, periods_per_year, v=vol)
            df  = append_row_df(df,vol,ret,spr,w)
            ax.scatter([vol], [ret], color="goldenrod", marker="o", label="EW portfolio")
            ax.legend()
        return df, ax
    else: 
        return df
    
    

    
from numpy.linalg import inv
def inverse_df(d):
    '''
    Inverse of a pd.DataFrame (i.e., inverse of dataframe.values)
    '''
    return pd.DataFrame( inv(d.values), index=d.columns, columns=d.index) 



def weigths_max_sharpe_ratio(covmat, mu_exc, scale=True):
    '''
    - mu_exc ===> بردار بازده مورد انتظار مازاد است (باید بردار ستونی به عنوان یک سری پانداسی باشد)
    - covmat ===> covariance N x N matrix as a pd.DataFrame
    '''
    w = inverse_df(covmat).dot(mu_exc)
    if scale:
        # normalize weigths
        w = w/sum(w) 
    return w
    
    

    
def get_ind_file(filetype="rets", nind=30, ew=False):
    '''
    - filetype: can be "rets", "nfirms", "size"
    - nind: 30 or 49
    - ew ===> if True, مجموعه داده های پرتفوهای با وزن یکسان بارگیری می شوند.
    '''
    if nind!=30 and nind!=49:
        raise ValueError("Expected either 30 or 49 number of industries")
    if filetype == "rets":
        portfolio_w = "ew" if ew==True else "vw" 
        name = "{}_rets" .format( portfolio_w )
        divisor = 100.0
    elif filetype == "nfirms":
        name = "nfirms"
        divisor = 1
    elif filetype == "size":
        name = "size"
        divisor = 1
    else:
        raise ValueError("filetype must be one of: rets, nfirms, size")
    ind = pd.read_csv("ind{}_m_{}.csv" .format(nind, name), index_col=0, parse_dates=True) / divisor
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    return ind    