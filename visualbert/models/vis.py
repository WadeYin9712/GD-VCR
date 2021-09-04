import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.ticker as ticker

def grounding_vis(attention_weights, texts, file_name, region, single_or_not):
    df = pd.DataFrame(attention_weights, columns=texts, index=texts)
    fig = matplotlib.pyplot.figure(figsize=(11,11))
    
    if single_or_not == "single":
        single = ""
    else:
        single = "_multiple"
    
    ax = fig.add_subplot(111)
    
    cax = ax.matshow(df, interpolation='nearest', cmap='hot_r')
    fig.colorbar(cax)
    
    tick_spacing = 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    
    fontdict = {'rotation': 60}
    ax.set_xticklabels([''] + list(df.columns), fontdict=fontdict)
    ax.set_yticklabels([''] + list(df.index))
    
    matplotlib.pyplot.savefig("../grounding_results_"+region+single+"/"+file_name)
        
