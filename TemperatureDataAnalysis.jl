############################################################################
#### Execute code chunks separately in VSCODE by pressing 'Alt + Enter' ####
############################################################################

using Statistics
using DataFrames
using GLM
using Plots
using GaussianProcesses
using GaussianRandomFields
using Extremes
using Optim
using JLD2
@load "/home/oestinmo/Dokumente/Vortraege/KIT_SummerSchool/TemperatureData.jld2"

##

no_times, no_stations = size(tempmean)

######################################################
########### Part I: Gaussian Modelling ###############
######################################################

tmean_mean = mean(tempmean, dims=1)
describe(tmean_mean)
tmean_std  = std(tempmean, dims=1)
describe(tmean_std)

##

scatter(
    vec(stations[:,"LON"]),
    vec(tmean_mean),
    xlabel="Longitude",
    ylabel="Mean Temperature")

##

scatter(
    vec(stations[:,"LAT"]),
    vec(tmean_mean),
    xlabel="Latitude",
    ylabel="Mean Temperature")

## set up model individually 

distancetolinear = function(betas)
   sum( (vec(tmean_mean) .- betas[1] - betas[2].*stations.LAT - betas[3].*stations.LON).^2 )
end

mu_model = optimize(distancetolinear, [0.0, 0.0, 0.0])

mu_pars = Optim.minimizer(mu_model)
mu_predicted = mu_pars[1] .+ mu_pars[2].*stations.LAT + mu_pars[3].*stations.LON

plot!(stations.LAT, mu_predicted)

## now the linear model

dataforlm  = DataFrame(LAT = repeat(vec(stations.LAT), inner=no_times),
                        LON = repeat(vec(stations.LON), inner=no_times),
                        MEAN = vec(tempmean))

lin_model = lm(@formula(MEAN ~ LAT + LON), dataforlm)

## now: set-up marginal model

stations_means = GLM.predict(lin_model, 
                            DataFrame(LAT=stations.LAT, LON=stations.LON))
stations_std  = mean(tmean_std)

## Calculate empirical correlation

emp_corr     = Array{Float64}(undef, no_stations, no_stations)
dist_matrix  = Array{Float64}(undef, no_stations, no_stations)
for i in 1:no_stations
  for j in 1:no_stations
    emp_corr[i,j] = cor(tempmean[:,i], tempmean[:,j])
    dist_matrix[i,j] = sqrt(  (stations.LON[i]-stations.LON[j])^2
                            + (stations.LAT[i]-stations.LAT[j])^2 )
  end
end    

## optimize covariance function

distancetokernel = function(logscale)
   myscale = exp(logscale)
   mykernelmat = Array{Float64}(undef, no_stations, no_stations)
   for i in 1:no_stations
     for j in 1:no_stations
       mykernelmat[i,j] = exp(- dist_matrix[i,j]/myscale)
     end
   end
   return(sqrt(sum( (mykernelmat-emp_corr).^2 )))    
end

cov_res = optimize(distancetokernel, -10.0, 10.0)

logscale = Optim.minimizer(cov_res)

## Interpolation: GP regression / Kriging

mZero = MeanZero()
kernel = Mat12Iso(logscale, log(stations_std))
gp = GP(transpose([stations.LON stations.LAT]),
        tempmean[10,:] - stations_means,
        mZero, kernel, -10)

p1 = plot(gp; title="Mean of GP")
p2 = plot(gp; var=true, title="Variance of GP", fill=true)    
scatter!(stations.LON, stations.LAT)             

## Conditional Simulation

xseq = 4.5:0.1:7.0
yseq = 51.0:0.1:53.0
x = transpose(Array(DataFrame(Iterators.product(xseq,yseq))))
samples = rand(gp, x, 5)
surface(x[1,:], x[2,:], samples[:,1])

## Simulate Gaussian Process

cov = CovarianceFunction(2, Exponential(exp(logscale)))
grf = GaussianRandomField(cov, CirculantEmbedding(), xseq, yseq)
contourf(grf)

######################################################
######### Part II: Modelling of Extremes #############
######################################################

gevparam = Array{Float64}(undef, no_stations, 3)
for i in 1:no_stations
  fm = gevfit(tempmax[:,i])
  gevparam[i,1] = location(fm)[1]
  gevparam[i,2] = scale(fm)[1]
  gevparam[i,3] = shape(fm)[1]
end  

describe(gevparam[:,1])
describe(gevparam[:,2])
describe(gevparam[:,3])

##
dataforlm  = DataFrame(LAT = repeat(vec(stations.LAT), inner=no_times),
                       LON = repeat(vec(stations.LON), inner=no_times),
                       MAX = vec(tempmax))
fm = gevfit(dataforlm, :MAX, locationcovid=[:LAT, :LON])