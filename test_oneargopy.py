############################################################################################################################
# Argo Functions

import time
from Argo import Argo

argo = Argo() # initialization

profiles = argo.select_profiles(lon_lim=[-127,-115],
                                 lat_lim=[32.5,45.5],
                                 start_date='2021-09-01',
                                 type='bgc')
argo.trajectories(list(profiles))
data = argo.load_float_data(profiles, 
                            variables=['TEMP', 'DOXY'])
argo.sections(data, 'TEMP', save_to='Plots')

start_time = time.time()
argo.sections(data, ['DOXY', 'DOXY_ADJUSTED'])
elapsed_time = time.time() - start_time
print(f'The time to plot doxy: {elapsed_time}\n')

print(f'Passing Nothing')
data = argo.load_float_data(5905105)
print(data)
data.to_csv('output_one.txt', encoding='utf-8', index=False, na_rep='nan')
print(f'\n\n')

print(f'Passing DOXY and CHLA')
data = argo.load_float_data([5904859, 5903807], variables=['DOXY', 'CHLA'])
print(data)
data.to_csv('output_two.txt', encoding='utf-8', index=False, na_rep='nan')
print(f'\n\n')

print(f'Passing TEMP')
data = argo.load_float_data([4903500, 5903611], variables=['TEMP'])
print(data)
data.to_csv('output_three.txt', encoding='utf-8', index=False, na_rep='nan')
print(f'\n\n')

print(f'Passing TEMP, DOXY, PRES')
data = argo.load_float_data([5904859, 5903807, 5906297], variables=['TEMP', 'DOXY', 'PRES'])
print(data)
data.to_csv('output_four.txt', encoding='utf-8', index=False, na_rep='nan')
print(f'\n\n')

print(f'Passing DOXY')
floats = argo.select_profiles(start_date='2024-05-01', end_date='2024-05-02', type='bgc') 
data = argo.load_float_data(floats, variables='DOXY')
print(data)
data.to_csv('output_five.txt', encoding='utf-8', index=False, na_rep='nan')
print(f'\n\n')

print(f'Passing DOXY')
floats = argo.select_profiles(start_date='2024-05-01', end_date='2024-05-02', type='bgc') 
data = argo.load_float_data(floats, variables=['TEMP', 'PSAL'])
print(data)
data.to_csv('output_five.txt', encoding='utf-8', index=False, na_rep='nan')
print(f'\n\n')

print('PROFILE INDEXES TEST')
floats = argo.select_profiles(start_date='2023-10-10', floats=5906297) 
data = argo.load_float_data(floats)
print(data)

profiles = argo.select_profiles(lon_lim=[-127,-115],
                                lat_lim=[32.5,45.5],
                                start_date='2021-09-01',
                                type='bgc')
print('trajectories from list of profiles')
argo.trajectories(list(profiles))

print('trajectories from dict of profiles')
argo.trajectories(profiles)

print('Trajectories Tests')
argo.trajectories(5905105)
argo.trajectories(5904859)
floats = [5905105, 5904859]
argo.trajectories(floats)

print('Testing dict passing')
profiles = argo.select_profiles([-170, -168], [20, 25], '2012-01-01', '2013-01-01')
argo.trajectories(profiles)
profiles = argo.select_profiles([-170, -168], [20, 25], '2012-01-01', '2013-01-01', outside='time')
argo.trajectories(profiles)

profiles = argo.select_profiles([100, 140], [30, 45])
argo.trajectories(list(profiles.keys()))

profiles = argo.select_profiles([0, 40])
argo.trajectories(list(profiles.keys()), save_to='Plots')

print('select_profiles, No Criteria Specified')
start_time = time.time()
argo.select_profiles()
elapsed_time = time.time() - start_time
print(f'This test took: {elapsed_time}\n')

print('Testing get by ocean basin:')
start_time = time.time()
profiles = argo.select_profiles(start_date='2012-01-01', end_date='2012-01-02', ocean='A')
argo.trajectories(profiles)
elapsed_time = time.time() - start_time
print(f'This test took: {elapsed_time}\n')

print('Now testing get by float id')
start_time = time.time()
argo.select_profiles(start_date='2012-01-01', end_date='2013-01-01', floats=[5903611, 5903802, 5903807])
elapsed_time = time.time() - start_time
print(f'This test took: {elapsed_time}\n')

start_time = time.time()
argo.select_profiles(floats=5903611)
elapsed_time = time.time() - start_time
print(f'This test took: {elapsed_time}\n')

start_time = time.time()
argo.select_profiles(floats=5903611)
elapsed_time = time.time() - start_time
print(f'This test took: {elapsed_time}\n')

start_time = time.time()
argo.select_profiles(floats=[4903500, 5903611])
elapsed_time = time.time() - start_time
print(f'This test took: {elapsed_time}\n')

print('Now testing outside functionality')
start_time = time.time()
print(f'OUTSIDE = NONE')
argo.select_profiles([-170, -168], [20, 25], '2012-01-01', '2013-01-01')
elapsed_time = time.time() - start_time
print(f'This test took: {elapsed_time}\n')

start_time = time.time()
print(f'OUTSIDE = TIME')
argo.select_profiles([-170, -168], [20, 25], '2012-01-01', '2013-01-01', outside='time')
elapsed_time = time.time() - start_time
print(f'This test took: {elapsed_time}\n')

start_time = time.time()
print(f'OUTSIDE = SPACE')
argo.select_profiles([-170, -168], [20, 25], '2012-01-01', '2013-01-01', outside='space')
elapsed_time = time.time() - start_time
print(f'This test took: {elapsed_time}\n')

start_time = time.time()
print(f'OUTSIDE = BOTH')
argo.select_profiles([-170, -168], [20, 25], '2012-01-01', '2013-01-01', outside='both')
elapsed_time = time.time() - start_time
print(f'This test took: {elapsed_time}\n')


print(f'Testing Min-Max VS Rectangle:')
print(f'Min-Max:')
start_time = time.time()
argo.select_profiles([-170, -168], [20, 25])
elapsed_time = time.time() - start_time
print(f'This test took: {elapsed_time}\n')

print(f'Rectangle:')
start_time = time.time()
argo.select_profiles([-168, -168, -170, -170], [20, 20, 25, 25])
elapsed_time = time.time() - start_time
print(f'This test took: {elapsed_time}\n')

print(f'Rectangular without dates:')
start_time = time.time()
argo.select_profiles([-170, -168], [20, 25])
elapsed_time = time.time() - start_time
print(f'This test took: {elapsed_time}\n')

print(f'Polygon with dates:')
start_time = time.time()
argo.select_profiles([38.21, 31.26, 29.77], [-74.8, -65.57, -80.16], '2013-01-01', '2020-01-01')
elapsed_time = time.time() - start_time
print(f'This test took: {elapsed_time}\n')

print(f'Lon only with dates:')
start_time = time.time()
argo.select_profiles(lon_lim=[-170, -168], start_date='2012-01-01', end_date='2014-01-01' )
elapsed_time = time.time() - start_time
print(f'This test took: {elapsed_time}\n')

print(f'Lat only without dates:')
start_time = time.time()
argo.select_profiles(lat_lim=[20, 25])
elapsed_time = time.time() - start_time
print(f'This test took: {elapsed_time}\n')

print(f'Dates only:')
start_time = time.time()
argo.select_profiles(start_date='2017-01-01', end_date='2019-12-31')
elapsed_time = time.time() - start_time
print(f'This test took: {elapsed_time}\n')
