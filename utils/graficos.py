# Reginaldo Ferreira
# import matplotlib
# matplotlib.use('GTK4Agg')  # or 'GTK4Cairo'
import matplotlib.pyplot as plt

# %config InlineBackend.figure_formats = ['svg']

# use 'dark_background', 'classic' or 'default'
# para ver todos os estilos: print(plt.style.available)
#%%
# bg_color = '#282C34' #--> one dark
# bg_color = '#2E3440' #--> nord
# bg_color = '#232731' #--> nord_Pro
# bg_color = '#ECEFF4' #--> nord_lPro Light
# bg_color = '#22272E' #--> Gnome Dimmed
# bg_color = '#242424' #--> Gnome Dark
# bg_color = '#FDF6E3' #--> solarized light
# bg_color = '#002B36' #--> solarized dark
bg_color = '#1d1d1d' #--> Adwaita Dark
# bg_color = '#F8F8F8' #--> Brackets Light Pro
# bg_color = '#FAFAFA' #--> Gnome Light (Atom One)
# bg_color = '#1D2021' #--> GruvBox-Hard
# bg_color = '#282828' #--> GruvBox-Medium
# bg_color = '#F9F5D7' #--> GruvBox-LGHT-Medium
# bg_color = '#F9F5D7' #--> GruvBox-LGHT-Hard

plt.style.use('dark_background')
# plt.style.use('default')


plt.rc('font', family='serif', size=15)
plt.rc('figure', facecolor=bg_color)
plt.rc('axes', facecolor=bg_color)
plt.rcParams["figure.figsize"] = (15,4)

#%%
