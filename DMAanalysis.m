Load data
addpath(genpath(pwd));

load battledata.mat

TQ = InflowData1.DatetimeCETCESTDDMMYYYYHHmm;
Q = table2array(InflowData1(:,2:11));
TW = WeatherData1.DatetimeCETCESTDDMMYYYYHHmm;
W = table2array(WeatherData1(:,2:5));