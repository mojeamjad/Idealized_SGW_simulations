def marker_color(df):
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors

    markers = ['o', 's', 'X', '^', 'D', 'P','1','4']  # Adjust for the number of unique L0
    tableau_colors = mcolors.TABLEAU_COLORS
    colors = list(tableau_colors.keys())[:5] +['c']  # Adjust for the number of unique Fr

    tnames=[]                    # test_names
    tstrength = ["    "]*len(df)              # test_strength based on Fr0
    
    for i in range (len(df)):
        # Decode string only if it is a bytes object
        if isinstance(df.at[i, 'Test_name'], bytes):
            df.at[i, 'Test_name'] = df.at[i, 'Test_name'].decode('utf-8')  # Fixing problem with undefined strings

        if df.at[i,'Test_name'][-1]=="1":
            df.at[i,'Test_name'] = df.at[i,'Test_name'][:-1]
        ##########################
        if df.at[i,'Test_name'][2:] == "vvlow":
            df.at[i,'color'] = colors[0]
        elif df.at[i,'Test_name'][2:] == "vlow":
            df.at[i,'color'] = colors[1]
        elif df.at[i,'Test_name'][2:] == "low":
            df.at[i,'color'] = colors[2]
        elif df.at[i,'Test_name'][2:] == "mid":
            df.at[i,'color'] = colors[3]               
        elif df.at[i,'Test_name'][2:] == "high":
            df.at[i,'color'] = colors[4]    
        else:
            df.at[i,'color'] = 'c'
        ################################    
        if df.at[i,'Test_name'] == "B3asc":
            df.at[i,'marker'] = markers[6]
        elif df.at[i,'Test_name'] == "B3rsg":
            df.at[i,'marker'] = markers[7]
        else:
            if df.at[i,'Test_name'][:2] == "B1":
                df.at[i,'marker'] = markers[0]
            elif df.at[i,'Test_name'][:2] == "B2":
                df.at[i,'marker'] = markers[1]
            elif df.at[i,'Test_name'][:2] == "B3":
                df.at[i,'marker'] = markers[2]
            elif df.at[i,'Test_name'][:2] == "B4":
                df.at[i,'marker'] = markers[3]
            elif df.at[i,'Test_name'][:2] == "B5":
                df.at[i,'marker'] = markers[4]
            elif df.at[i,'Test_name'][:2] == "B6":
                df.at[i,'marker'] = markers[5]
                index_B6 = i
            ##################################  
            tnames.append(df.at[i,'Test_name'][:2])
            tstrength[i] = df.at[i,'Test_name'][2:]
        ##################################
    tnames = sorted(list(set(tnames))) # Removing duplicates by first changing to a set and then back to a list
    if len(df) > index_B6+1:
        markerss = list(set(df.loc[:,'marker'][:-2]))
    else:
        markerss = list(set(df.loc[:,'marker'][:]))
    marks2 = [item for item in markers if item in markerss]

    assert len(tnames) == len(marks2), f"Series sizes do not match: {len(tnames)} != {len(marks2)}"
    
    # Create legend entries for markers (test names)
    marker_legend = [
        Line2D([0], [0], marker=marker, color='w', label=f'Test: {name}', 
               markerfacecolor='black', markersize=10, linestyle='None',alpha=0.6)
        for name, marker in zip(tnames, marks2)]
    
    tstrength = sorted(list(set(tstrength[:index_B6])), reverse=True)# Removing duplicates by first changing to a set and then back to a list
    tstrength[2], tstrength[3] = tstrength[3], tstrength[2]
    
    # Create legend entries for colors (Fr labels)
    color_legend = [
        mpatches.Patch(color=color, label=f'{label}')
        for label, color in zip(tstrength, colors)]
        
    # Combine marker and color legends
    legend_elements = color_legend + marker_legend 

    # Adding "B3asc" and "B3rsg"
    if len(df) > index_B6+1:
        marker_legend2 = [
            Line2D([0], [0], marker=marker, color='c', label=f'{name}', 
               markerfacecolor='c', markersize=10, linestyle='None')
            for name, marker in zip(
                ["B3asc","B3rsg"], markers[-2:])]
        legend_elements = color_legend  + marker_legend + marker_legend2
        
        
    
    return df, tnames, tstrength, legend_elements

########################################################################    
def s_transfer(ft,dt):
    
    import numpy as np
    twopi = 2*np.pi
    num = len(ft)
    stf = np.zeros((num,num))+complex(0)
    f1 =1/num
    ff = np.linspace(0, num,num)*f1
    tt = np.linspace(0, num,num)
    c = 1.0   #window
    f_f = np.fft.fft(ft)

    # plt.plot(ff[1:151]*twopi/0.5,abs(f_f[1::]))

    for fInd in range(1,num):
        f = ff[fInd]
        wgt = np.exp(-2*(np.pi*c*ff/f)**2)
        fct = np.roll(f_f, -fInd)
        scr = fct * wgt
        scr = np.fft.ifft(scr)
        stf[:,fInd] = scr 
        
    real_ff = ff[1:int(num/2)+1]*twopi/dt
    cff = abs(np.fft.rfft(ft)[1::])
    
    return abs(stf.T),cff,real_ff
########################################################################    
def S2(lon,time,height,u0,m_u,fig,ax,title_name=None):
    import numpy as np
    from matplotlib.ticker import (MultipleLocator,  AutoMinorLocator, ScalarFormatter)
    import matplotlib as mlt
    import matplotlib.pyplot as plt
    plt.rc('axes', linewidth=2)
    plt.rcParams['font.size'] = '20'
    
    distance = lon*110 # zonal distance (km)
    xv, yv = np.meshgrid(distance, height)
    u_p = u0 - m_u[:, np.newaxis]

    # lim = abs(u0).max()*2/3 # colormap max range : Supplementary
    lim = abs(u0).max()/3 # colormap max range    : Main article

    levels = np.linspace(-1,1,41)*lim
    norm = mlt.colors.Normalize(vmin=levels[0], vmax=levels[-1], clip=False)
    
    #fig,ax = plt.subplots(figsize=(7,10))
    
    pl = ax.contourf(xv,yv,u_p,levels,origin='upper',cmap='bwr',norm=norm)
    
    ax.set_ylim(0,150)
    ax.tick_params(axis='both', which='major',  length=8, width=3, direction ='in'
                   , top=True, right=True)
    ax.tick_params(axis='both', which='minor',  length=4, width=1.5, direction ='in'
               , top=True, right=True)
    ax.yaxis.set_major_locator(MultipleLocator(30))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    if distance.max() < 100:
        maj_loc = 50
        min_loc = 10
    elif 100 < distance.max() < 500:        
        maj_loc = 100
        min_loc = 50
    elif 500 < distance.max() < 1000:       
        maj_loc = 200
        min_loc = 100
    else:     
        maj_loc = 1000
        min_loc = 500                
    ax.xaxis.set_major_locator(MultipleLocator(maj_loc)) 
    ax.xaxis.set_minor_locator(MultipleLocator(min_loc))
    for label in ax.get_xticklabels():
        label.set_rotation(-30)
        
    title_time = f"(t={round(time, 1)} day, y=0.0)"
    ax.text(
        0.3, 1.03,  # Same position for alignment
        title_name, fontweight='bold',
        fontsize=19,  # Larger font size for the main title
        ha='right', va='center', transform=ax.transAxes  # Use relative coordinates
    )
    ax.text(
        0.5, 1.03,  # Position above the plot (relative to axes)
        f"{title_time}", fontsize=15,  # Font size for the entire string
        ha='left', va='center', transform=ax.transAxes  # Use relative coordinates
    )
    cax = ax.inset_axes([0.05, 0.07, 0.2, 0.02])
    cbar = fig.colorbar(pl,cax=cax, orientation='horizontal' ,ticks=[int(levels[0]),0, int(levels[-1])])
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.set_xticklabels([int(levels[0]),0, int(levels[-1])])  # horizontal colorbar
    cbar.ax.set_title("$u'$ (m/s)", fontsize=16)
 
########################################################################  
def S3(time,height,std_u,ax=None,title_name=None):

    import numpy as np
    from matplotlib.ticker import (MultipleLocator,  AutoMinorLocator, ScalarFormatter)
    import matplotlib as mlt
    import matplotlib.pyplot as plt
    plt.rc('axes', linewidth=2)
    plt.rcParams['font.size'] = '20'
    
    #fig,ax = plt.subplots(figsize=(7,10))
    ax.plot(height,std_u)
    
    ax.set_xlim(0,150)
    ax.set_ylim(0,None)
    ax.tick_params(axis='both', which='major',  length=8, width=3, direction ='in'
                   , top=True, right=True)
    ax.tick_params(axis='both', which='minor',  length=4, width=1.5, direction ='in'
               , top=True, right=True)
    ax.xaxis.set_major_locator(MultipleLocator(30))    
    ax.xaxis.set_minor_locator(MultipleLocator(10))

    if std_u.max() < 1:
        maj_loc = 0.1
        min_loc = 0.05
    elif 1 < std_u.max() < 5:        
        maj_loc = 0.5
        min_loc = 0.1
    elif 5 < std_u.max() < 12:       
        maj_loc = 1
        min_loc = 0.5
    else:     
        maj_loc = 3
        min_loc = 1                
    ax.yaxis.set_major_locator(MultipleLocator(maj_loc))
    ax.yaxis.set_minor_locator(MultipleLocator(min_loc))
    
        
    title_time = f"(t={round(time, 1)} day)"
    ax.text(
        0.3, 1.03,  # Same position for alignment
        title_name, fontweight='bold',
        fontsize=19,  # Larger font size for the main title
        ha='right', va='center', transform=ax.transAxes  # Use relative coordinates
    )
    ax.text(
        0.5, 1.03,  # Position above the plot (relative to axes)
        f"{title_time}", fontsize=15,  # Font size for the entire string
        ha='left', va='center', transform=ax.transAxes  # Use relative coordinates
    )
########################################################################    
def S4(lon,time,height,u0,m_u,ax=None,title_name=None):
    
    import numpy as np
    from matplotlib.ticker import (MultipleLocator,  AutoMinorLocator, ScalarFormatter)
    import matplotlib as mlt
    import matplotlib.pyplot as plt
    plt.rc('axes', linewidth=2)
    plt.rcParams['font.size'] = '20'
    
    distance = lon*110 # zonal distance (km)

    r_u = abs(u0 - m_u)/ m_u
    #fig,ax = plt.subplots(figsize=(7,10))
    ax.plot(height,r_u)
    
    ax.set_xlim(0,150)
    ax.set_ylim(-0.01,1.1)
    ax.tick_params(axis='both', which='major',  length=8, width=3, direction ='in', top=True, right=True)
    ax.tick_params(axis='both', which='minor',  length=4, width=1.5, direction ='in', top=True, right=True)
    ax.xaxis.set_major_locator(MultipleLocator(30))    
    ax.xaxis.set_minor_locator(MultipleLocator(10))
              
    ax.yaxis.set_major_locator(MultipleLocator(0.2)) 
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))    
        
    title_time = f"(t={round(time, 1)} day, y=0.0)"
    ax.text(
        0.3, 1.03,  # Same position for alignment
        title_name, fontweight='bold',
        fontsize=19,  # Larger font size for the main title
        ha='right', va='center', transform=ax.transAxes  # Use relative coordinates
    )
    ax.text(
        0.5, 1.03,  # Position above the plot (relative to axes)
        f"{title_time}", fontsize=15,  # Font size for the entire string
        ha='left', va='center', transform=ax.transAxes  # Use relative coordinates
    )
#################################################
def S5(time,height,a_x2,ax=None,title_name=None):

    import numpy as np
    from matplotlib.ticker import (MultipleLocator,  AutoMinorLocator, ScalarFormatter)
    import matplotlib as mlt
    import matplotlib.pyplot as plt
    plt.rc('axes', linewidth=2)
    plt.rcParams['font.size'] = '20'
    
    #fig,ax = plt.subplots(figsize=(7,10))
    ax.plot(height,a_x2)
    
    ax.set_xlim(5,145)
    # ax.set_ylim(0,None)
    ax.tick_params(axis='both', which='major',  length=8, width=3, direction ='in'
                   , top=True, right=True)
    ax.tick_params(axis='both', which='minor',  length=4, width=1.5, direction ='in'
               , top=True, right=True)
    ax.xaxis.set_major_locator(MultipleLocator(20))    
    ax.xaxis.set_minor_locator(MultipleLocator(5))

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))  # Use scientific notation for large/small numbers
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.get_offset_text().set_position((-0.1, 0))  # Adjust (x, y) position
    
    title_time = f"(t={round(time, 1)} day, x=0.0, y=0.0)"
    ax.text(
        0.4, 1.03,  # Same position for alignment
        title_name, fontweight='bold',
        fontsize=19,  # Larger font size for the main title
        ha='right', va='center', transform=ax.transAxes  # Use relative coordinates
    )
    ax.text(
        0.5, 1.03,  # Position above the plot (relative to axes)
        f"{title_time}", fontsize=15,  # Font size for the entire string
        ha='left', va='center', transform=ax.transAxes  # Use relative coordinates
    )
########################################################################    
def S6(fig, spec, lon,time,z,uu,powerd, spectrum, freq,title_name):
    import numpy as np
    from matplotlib import pyplot as plt
    import matplotlib as mlt
    from matplotlib.ticker import MultipleLocator,LogLocator, FixedLocator
    from matplotlib.ticker import ScalarFormatter,NullFormatter
    from matplotlib import gridspec
    plt.rcParams.update({'font.size': 18})
    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["axes.labelweight"] = "normal"
    params = {'mathtext.default': 'regular' }
    plt.rcParams['axes.linewidth'] = 3
    
    y_formatter = ScalarFormatter()
    distance = lon*110 # zonal distance (km)
    mid_ind_lon = np.argmin(np.abs(distance))
    xrange = 60
    
    cone=np.zeros_like((len(distance),len(freq)))
    cone=np.sqrt(2)/freq*(6/(2*np.pi))
    
    axu = fig.add_subplot(spec[0, 0]) # Top-left panel
    axs = fig.add_subplot(spec[1, 0]) # Bottom-left panel
    axm = fig.add_subplot(spec[1, 1]) # Bottom-right panel
    
    if title_name[:2] =='B1': maj_loc = 0.5 
    if title_name[:2] =='B2': maj_loc = 0.2 
    if title_name[:2] =='B3': maj_loc = 0.1 
    if title_name[:2] =='B4': maj_loc = 0.05
    if title_name[:2] =='B5': maj_loc = 0.02
    if title_name[:2] =='B6': maj_loc = 0.01 
    min_loc = maj_loc/4
    ################
    
    # axu.set_title("\ny=0,z="+str(round(z))+",t="+str(round(time,2))+"(day)",y=1.1,x=0.6, pad=+14)
    axu.plot(distance,uu)
    axu.set_xlim([distance.min(), distance.max()])
    axu.set_xticklabels([])
    
    title_time = f"(t={round(time, 1)} day, y=0.0, z={z} )" 
    axu.text(
        0.2, 1.13,  # Same position for alignment
        title_name, fontweight='bold',
        fontsize=19,  # Larger font size for the main title
        ha='right', va='center', transform=axu.transAxes  # Use relative coordinates
    )
    axu.text(
        0.4, 1.13,  # Position above the plot (relative to axes)
        f"{title_time}",
        fontsize=15,  # Font size for the entire string
        ha='left', va='center', transform=axu.transAxes  # Use relative coordinates
    )
    ################    
    
    axm2 = axm.twiny()
    
    mean_m =  np.mean(powerd[:,mid_ind_lon-xrange:mid_ind_lon+xrange],axis=1) # 
    mean_r =  np.mean(powerd[:,mid_ind_lon+xrange:-5],axis=1) # 
    mean_l =  np.mean(powerd[:,5:mid_ind_lon-xrange],axis=1) 
        
    p1, = axm2.plot(mean_l,freq,'blue', label='P1')
    p2, = axm2.plot(mean_m,freq,'red', label='P2')
    p3, = axm2.plot(mean_r,freq,'green', label='P3')
    p4, = axm.plot(spectrum[:],freq[:],'--k', label='FFT')
    
    axm2.set_xticklabels([])

    max_y = freq[-1]/4
    axm.set_ylim([0, max_y])
    # axm.set_xlim([-0.1, None])
    axm.set_yticklabels([])
    axm.set_xticklabels([])
    axm.yaxis.set_major_locator(MultipleLocator(maj_loc))
    axm.yaxis.set_minor_locator(MultipleLocator(min_loc))
    axm.tick_params(axis='y', which='major', labelsize=24, length=15, width=2, left=False, right=True,direction= 'in')
    axm.tick_params(axis='y', which='minor', length=5, width=1.5, left=False, right=True, direction= 'in')
    
    # Add both legends to the plot manually
    additional_legend = axm.legend(handles=[
        plt.Line2D([0], [0], linestyle=p1.get_linestyle(), color=p1.get_color(), label='P1'),
        plt.Line2D([0], [0], linestyle=p2.get_linestyle(), color=p2.get_color(), label='P2'),
        plt.Line2D([0], [0], linestyle=p3.get_linestyle(), color=p3.get_color(), label='P3'),
        plt.Line2D([0], [0], linestyle=p4.get_linestyle(), color=p4.get_color(), label='FFT'),        
        ], loc="upper center", fontsize=13, bbox_to_anchor=(0.55, 1.25))        
    axm.add_artist(additional_legend)  # Add the line legend manually
    
    axm.set_xlabel('Intensity',fontsize=14)
    ################

    max_l = round(np.max(powerd),3)  # activate for the main article
    min_l = round(max_l / 4, 3) if max_l < 1 else round(max_l / 3.2, 2)
    # max_l = round(np.max(powerd),2) # activate for the Supplementary
    # min_l = round(max_l / 4, 2) if max_l < 1 else round(max_l / 3.2, 1)
    levels = np.linspace(min_l,max_l,11) 
    norm = mlt.colors.Normalize(vmin=levels[0], vmax=levels[-1], clip=False)    
    
    pl = axs.contourf(distance,freq,powerd,norm=norm, levels=levels,alpha=0.8)
        
    axs.set_ylim([0, max_y])
    axs.set_xlim([distance.min(), distance.max()])
    axs.tick_params(axis='both', which='major', labelsize=20, length=10, width=2,
                    top = True, bottom=True, left=True, right=True, direction= 'in')
    axs.tick_params(axis='both', which='minor', length=5, width=1.5, direction= 'in')
    axs.yaxis.set_minor_locator(MultipleLocator(min_loc))
    axs.yaxis.set_major_locator(MultipleLocator(maj_loc))
    # axs.xaxis.set_minor_locator(MultipleLocator(1))
    # axs.xaxis.set_major_locator(MultipleLocator(2))
    #cone of influence
    axs.fill_betweenx(freq,(max(cone)-cone)/(max(cone))*max(distance),max(distance),alpha=0.4,hatch="--", color='gray')
    axs.fill_betweenx(freq,min(distance),(cone-max(cone))/(max(cone))*max(distance),alpha=0.4,hatch="--", color='gray')
    
    cax = axs.inset_axes([0.07, 0.9, 0.4, 0.025])
    cbar = fig.colorbar(pl,cax=cax, orientation='horizontal',ticks=[levels[0],levels[int(len(levels)/2)],levels[-1]] )
    cbar.ax.tick_params(labelsize=17)
    cbar.ax.set_title("Intensity", fontsize=17)
    cbar.ax.set_xticklabels([levels[0],round(levels[int(len(levels)/2)],3), levels[-1]])  # horizontal colorbar
   
    return axs,axu
    
########################################################################    
def S7(fig, spec, height,time,x,uu,powerd,spectrum, freq,title_name=None):
    import numpy as np
    from matplotlib import pyplot as plt
    import matplotlib as mlt
    from matplotlib.ticker import MultipleLocator,LogLocator, FixedLocator
    from matplotlib.ticker import ScalarFormatter,NullFormatter
    from matplotlib import gridspec
    plt.rcParams.update({'font.size': 18})
    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["axes.labelweight"] = "normal"
    params = {'mathtext.default': 'regular' }
    plt.rcParams['axes.linewidth'] = 3
    y_formatter = ScalarFormatter()
    
    cone=np.zeros_like((len(height),len(freq)))
    cone=np.sqrt(2)/freq*(6/(2*np.pi))
    
    axu = fig.add_subplot(spec[0, 0]) # Top-left panel
    axs = fig.add_subplot(spec[1, 0]) # Bottom-left panel
    axm = fig.add_subplot(spec[1, 1]) # Bottom-right panel
        
    u_r = round(uu[-1]) # background wind (m/s)
    # print(u_r)
    if u_r < 8:       max_y = 10.1 ; maj_loc = 2   # light breeze  u0~5
    if 8  < u_r < 15: max_y = 5.05 ; maj_loc = 1   # fresh breeze  u0~10
    if 15 < u_r < 25: max_y = 3.02 ; maj_loc = 0.5 # fresh gale    u0~20
    if u_r >  30:     max_y = 1.52 ; maj_loc = 0.2 # hurricane     u0~40
    ################
    
    # axu.set_title("\ny=0,z="+str(round(z))+",t="+str(round(time,2))+"(day)",y=1.1,x=0.6, pad=+14)
    axu.plot(height,uu)
    axu.set_xlim([0, 151])
    axu.set_xticklabels([])
    
    title_time = f"(t={round(time, 1)} day, y=0.0, x={x}km)" 
    axu.text(
        0.2, 1.13,  # Same position for alignment
        title_name, fontweight='bold',
        fontsize=19,  # Larger font size for the main title
        ha='right', va='center', transform=axu.transAxes  # Use relative coordinates
    )
    axu.text(
        0.4, 1.13,  # Position above the plot (relative to axes)
        f"{title_time}",
        fontsize=15,  # Font size for the entire string
        ha='left', va='center', transform=axu.transAxes  # Use relative coordinates
    )
    ################    
    
    axm2 = axm.twiny()

    index20  = np.argmin(np.abs(height - 20.0))
    index80  = np.argmin(np.abs(height - 80.0))
    index100 = np.argmin(np.abs(height - 100.0))
    index120 = np.argmin(np.abs(height - 120.0))
    
    mean_s3 =  np.mean(powerd[:,index120:index100],axis=1) # # 100:120 km
    mean_s2 =  np.mean(powerd[:,index100:index80],axis=1) # # 80:100 km
    mean_s1 =  np.mean(powerd[:,index80:index20],axis=1)   # # 0:80   km
        
    s1, = axm2.plot(mean_s1,freq,'blue', label='S1')
    s2, = axm2.plot(mean_s2,freq,'red', label='S2')
    s3, = axm2.plot(mean_s3,freq,'green', label='S3')
    p4, = axm.plot(spectrum[:],freq[:],'--k', label='FFT')
    
    axm2.set_xticklabels([])

    axm.set_ylim([0, max_y])
    # axm.set_xlim([-0.1, None])
    axm.set_yticklabels([])
    axm.set_xticklabels([])
    axm.tick_params(axis='y', which='major', length=15, width=2, left=False, right=True,direction= 'in')
    axm.tick_params(axis='y', which='minor', length=5, width=1.5, left=False, right=True, direction= 'in')
    axm.yaxis.set_major_locator(MultipleLocator(maj_loc))
    axm.yaxis.set_minor_locator(MultipleLocator(maj_loc/5))
    
    # Add both legends to the plot manually
    additional_legend = axm.legend(handles=[
        plt.Line2D([0], [0], linestyle=s1.get_linestyle(), color=s1.get_color(), label='S1'),
        plt.Line2D([0], [0], linestyle=s2.get_linestyle(), color=s2.get_color(), label='S2'),
        plt.Line2D([0], [0], linestyle=s3.get_linestyle(), color=s3.get_color(), label='S3'),
        plt.Line2D([0], [0], linestyle=p4.get_linestyle(), color=p4.get_color(), label='FFT'),        
        ], loc="upper center", fontsize=13, bbox_to_anchor=(0.55, 1.25))        
    axm.add_artist(additional_legend)  # Add the line legend manually
    
    axm.set_xlabel('Intensity',fontsize=14)
    # ################

    max_l = round(np.max(powerd),2)
    min_l = round(max_l / 4, 3) if max_l < 1 else round(max_l / 3.2, 2)
    levels = np.linspace(min_l,max_l,11) 
    norm = mlt.colors.Normalize(vmin=levels[0], vmax=levels[-1], clip=False)    
    
    pl = axs.contourf(height,freq,powerd,norm=norm, levels=levels,alpha=0.8)
        
    axs.set_ylim([0, max_y])
    axs.set_xlim([0, 151])
    axs.tick_params(axis='both', which='major', labelsize=20, length=10, width=2, top = True, bottom=True, left=True, right=True, direction= 'in')
    axs.tick_params(axis='both', which='minor', length=5, width=1.5, direction= 'in')
    axs.yaxis.set_minor_locator(MultipleLocator(maj_loc/5))
    axs.yaxis.set_major_locator(MultipleLocator(maj_loc))
    axs.xaxis.set_minor_locator(MultipleLocator(5))
    axs.xaxis.set_major_locator(MultipleLocator(20))
    # #cone of influence
    axs.fill_betweenx(freq,(-cone+(max(cone)*2))/(max(cone)*2)*max(height),max(height),alpha=0.4,hatch="--", color='gray')
    axs.fill_betweenx(freq,0,cone/(max(cone)*2)*max(height),alpha=0.4,hatch="--", color='gray')
    axs.fill_betweenx(freq,120,150,alpha=0.1, color='gray') # sponge layer
    
    cax = axs.inset_axes([0.07, 0.9, 0.4, 0.025])    
    cbar = fig.colorbar(pl,cax=cax, orientation='horizontal',ticks=[levels[0],levels[int(len(levels)/2)],levels[-1]] )
    cbar.ax.tick_params(labelsize=17)
    cbar.ax.set_title("Intensity", fontsize=17)
    cbar.ax.set_xticklabels([levels[0],round(levels[int(len(levels)/2)],2), levels[-1]])  # horizontal colorbar

   
    return axs,axu
########################################################################    
    
