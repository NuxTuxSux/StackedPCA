using PyCall
PCA = pyimport("sklearn.decomposition").PCA



mutable struct Layer
    kernelshape :: Tuple{Int, Int}
    stride :: Tuple{Int, Int}
    nfilters :: Int
    kernel :: PyObject
    Layer(kernelshape, stride, nfilters) = (
        kernel = PCA(n_components = nfilters);
        new(kernelshape, stride, nfilters, kernel)
    )
    # Forse non è il modo migliore di fare questa cosa
    Layer(;kernelshape, stride, nfilters) = Layer(kernelshape, stride, nfilters)
end


function fit(layer::Layer, tensor::Array{Float64,4})
    # tensor = (nsamples, channels, width, height)
    kh, kw = layer.kernelshape
    sh, sw = layer.stride
    nsamples, channels, height, width = size(tensor)
    # muovo la kernel window di tot stride e concateno tutto lungo la prima dimensione
    data = vcat([reshape(tensor[:, :, i:(i+kh-1), j:(j+kw-1)],(nsamples,:)) for i ∈ 1:sh:(height-kh+1), j ∈ 1:sw:(width-kw+1)]...)
    layer.kernel.fit(data)
end

function transform(layer::Layer, tensor::Array{Float64,4})
    # tensor = (nsamples, channels, width, height)
    kh, kw = layer.kernelshape
    sh, sw = layer.stride
    nsamples, channels, height, width = size(tensor)
    newh, neww = length(1:sh:(height-kh+1)), length(1:sw:(width-kw+1))
    res = zeros(Float64,(nsamples, layer.nfilters, newh, neww))
    for n in 1:nsamples
        for i in 1:newh
            for j in 1:neww
                res[n,:,i,j] = layer.kernel.transform(reshape(tensor[n,:,i:(i+kh-1),j:(j+kw-1)],(1,:)))
            end
        end
    end
    res
end
    