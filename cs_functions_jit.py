from jax import numpy as jnp
import jax
from pennylane import numpy as np
import numpy as onp
from sklearn.metrics import r2_score,mean_squared_error


@jax.jit
def eval_cs_real(x,Ccos,Csin,Omega):
    #evaluate cs-fourier series expanded to cos and sin 
    # x: input data
    # Ccos,Csin: Coeffientes for cos and sin (real valued)
    # Omega: frequency spectrum
    #Output: 
    #y : evaluated fourier series
    num_freq = Omega.shape[0]
    if len(x.shape) < 2:
        y = jnp.zeros((1,x.shape[0])).astype('float64')#.astype('complex128')
    else:
        y = jnp.zeros(x.shape).astype('float64')#.astype('complex128')


    def body_fun(i,y):
        y += jnp.outer(Ccos[i,:],jnp.cos(jnp.dot(Omega[i,:],x.T))).T + jnp.outer(Csin[i,:],jnp.sin(jnp.dot(Omega[i,:],x.T))).T 
        return y
    y = jax.lax.fori_loop(0, num_freq, body_fun, y)
    return y

def compute_fourier_coeff_fft(model_qnn_q,Oflat,Xft,shape,opt_params,linear_pre_inverse):
    # Compute the coefficients of the fourier series with FFT. Only works for uniform spectrum
    # Input: 
    #    model_qnn_q: quantum part of qnn model (i.e. maps from R^d to R^d, without classical, affine linear tranformation). with opt_params from training
    # Oflat: Frequency spectrum (flattened)
    # Xft: points for FFT, correspond to frequency spectrum. Not flat
    # shape: shape of frequency spectrum
    # linear_pre_inverse: for preprocessing: This is here just the identity
    Yft_flat1 = np.zeros(Xft.shape)
    d = Yft_flat1.shape[1]

    #Eval cs  + for whole spectrum
    batch_size = 500
    
    no_batches_test = int(np.floor(Yft_flat1.shape[0] / batch_size)) 
    for kk in range(0, no_batches_test-1):
        ins_batch = Xft[kk*batch_size:(kk+1)*batch_size,:]
        outs_batch= np.array(model_qnn_q(opt_params,linear_pre_inverse(ins_batch))).T
        Yft_flat1[kk*batch_size:(kk+1)*batch_size,:] = outs_batch
 
    ins_batch = Xft[(no_batches_test-1)*batch_size:,:]
    outs_batch= np.array(model_qnn_q(opt_params,linear_pre_inverse(ins_batch))).T  
    Yft_flat1[(no_batches_test-1)*batch_size:,:]= outs_batch
    #print('done') 
    Yft_flat = jnp.asarray(Yft_flat1)


    Yft = onp.ascontiguousarray(Yft_flat).reshape(*shape) #reshape to size expected in ifftn
    Cft_np = np.fft.ifftn(Yft,axes = range(d)) 
    #Cftflat_np = Cft_np.reshape(-1,d)

    return Cft_np


def find_largest_indices_real_trunc(shape,Omegafft,Cft,d,n0,trunc_frequencies):
    # For output Omegafft and Cft of np.fft.ifftn, truncate the frequency spectrum (flattend) to number_samples largest coefficients 
    # and compute the coefficients for expanded fourier series into cosine/sin
    # Input:
    # shape: shape of Frequency spectrum for fftn
    # Omegafft. array(shape = shape) Frequency spectrum 
    # Cft: coefficients for Omegafft, output of ifftn
    # d: dimension of input features/number of qubits
    # n0: number of encoding gates
    # trunc_frequenices: list of integers, at which Coefficients are sorted, s.t. [0,trunc_frequencies[0]] are the largest coefficients and so on.
    # Output: 
    # Omega: Truncated frequency (flat, i.e. shape (number_frequencies x input dimension)
    # Ccos & Csin: Coefficients (real valued) of cosine + sine terms in fourier series 


    Cftflat = Cft.reshape(-1,d)
   
    shape2 = (2*n0+1,)*d
    Ccos = np.zeros(shape) 
    Csin= np.zeros(shape)
    active_freq = np.zeros(shape) # to control which frequencies where already sampled (only OMega, not -omega)
    cntr = 0
    
    Oflat = Omegafft.reshape(-1,d)
    C_norm = np.linalg.norm(Cftflat,axis = 1)
    indices = np.argpartition(-C_norm,trunc_frequencies)[:max(trunc_frequencies)] 
    #indices = np.argsort(-C_norm)[:max(trunc_frequencies)] # incase we want the coefficients in descending order, not just partitioned. but this is slower
    #insert zero frequency!
    indices = list(indices)
    indices.insert(0,0)
    
    Omega = []
    Ccos = []
    Csin = []
    for i0 in indices: #          
        A2 = np.zeros(((2*n0+1)**d,))
        A2[i0] = 1.
        #find i0th frequencies in shape2-shaped tensor (output of ifftn)
        A = A2.reshape(shape2)
        idx = np.nonzero(A)
        if (np.allclose(Omegafft[*idx,:], np.zeros((1,d))))&(active_freq[*idx,0]==0): #zero frequency
            active_freq[*idx,0] =1 #to mark that we have been here already
            cplus = Cft[*idx,:] 
            if not np.linalg.norm(np.imag(cplus)) < 1e-7:
                print(np.imag(cplus))
                raise ValueError('Input not real function')
            aomega = np.real(cplus)
            bomega = np.real(np.zeros(cplus.shape))
            Omega.append(Oflat[i0,:])
            Ccos.append(np.squeeze(aomega))
            Csin.append(np.squeeze(bomega))
        else:
            idx_neg = tuple((-i)%n for i,n in zip(idx,shape2)) #negative frequency

            #Check if symmetry of frequency is computed correctly
            if np.linalg.norm(Omegafft[*idx_neg,:] + Omegafft[*idx,:])>0:
                raise ValueError('Symmetry not correct.')
            if not ((active_freq[*idx,0] == 1) or (active_freq[*idx_neg,0] == 1)): #we havent been here already
                cplus = Cft[*idx,:] 
                cminus = Cft[*idx_neg,:]
                aomega = 2*np.real(cplus)
                bomega = 2*np.imag(cplus) # (No minus sign)
                
                Omega.append(Oflat[i0,:])
                Ccos.append(np.squeeze(aomega))
                Csin.append(np.squeeze(bomega))
                if not np.all( np.allclose(np.conjugate(cplus),cminus)):
                    raise ValueError('Coefficients not conjugate.') #there must be something wrong with computation of coefficients or output functions is not real valued
                else:
                    cntr += 1   
                active_freq[*idx,0] =1 
                active_freq[*idx_neg,0] = 1
    return np.array(Omega),np.asarray(Ccos),np.asarray(Csin) 


        

def eval_trunc_cs(Omegacutfft,Ccos,Csin, trunc_frequencies, test_input, test_output,indsIII,no_testing_data,no_test_samples_to_evaluate,post_processing,opt_params):
    # evluation of truncated fourier series. each segment is evaluated by it self, than added later to get the differently truncated series. Last step is post processing. 
    # Input: 
    # Omegacutfft: truncated frequencie spectrum, sorted accoring to trunc_frequenices. Ccos, Csin corresponding coefficients of frequency spectrum
    # Testing data: test_inputs, will only be evaluated at indsIII (otherwise 0), no_testing_data: length of entire test data, no_test_samples_to_evaluate: length of test data to evaluate (acc to indsIII)
    # post_processing: including mapping from R^d to R^1 (which needs opt_params from qml model

    # Output: MSE and R2 score to testing datas MSE and R2 score to testing data
    
    L = len(trunc_frequencies)
    pred_output = np.zeros((test_output.shape[0],Ccos.shape[1],L)).astype('float64')
    test_addition = np.zeros((L,))
    for i0 in range(L):
        #slice frequencies
        if i0 == 0: #witht zero
            Omegacutfft_slice = Omegacutfft[0:trunc_frequencies[0],:]
            Ccos_slice = Ccos[0:trunc_frequencies[0],:]
            Csin_slice = Csin[0:trunc_frequencies[0],:]
        elif i0 == (L-1):
            Omegacutfft_slice = Omegacutfft[trunc_frequencies[i0-1]:,:]
            Ccos_slice = Ccos[trunc_frequencies[i0-1]:,:]
            Csin_slice = Csin[trunc_frequencies[i0-1]:,:]
        else:
            Omegacutfft_slice = Omegacutfft[trunc_frequencies[i0-1]:trunc_frequencies[i0],:]
            Ccos_slice = Ccos[trunc_frequencies[i0-1]:trunc_frequencies[i0],:]
            Csin_slice = Csin[trunc_frequencies[i0-1]:trunc_frequencies[i0],:]
        pred_test_output_slice = eval_series_slice(Ccos_slice,Csin_slice, Omegacutfft_slice,test_input,indsIII, no_testing_data, no_test_samples_to_evaluate)


        for kk in range(i0,L):
            pred_output[:,:,kk] +=  np.copy(pred_test_output_slice) 

        

    
    mse_output = np.zeros((L,))
    r2_output = np.zeros((L,))
    for kk in range(L):
        outs_batch = post_processing(np.squeeze(pred_output[indsIII,:,kk]),opt_params)
        mse_output[kk] = mean_squared_error(test_output[indsIII],outs_batch)
        r2_output[kk] = r2_score(test_output[indsIII],outs_batch)

    return mse_output, r2_output



def eval_series_slice(Ccos,Csin,Omegacutfft,test_inputs,indsIII, no_testing_data, no_test_samples_to_evaluate):
    #Eval cs on a slice of a fourier series on the entire test data set(no postprocessing here!) (test_inputs). Evluation only on indsIII, see L. Pastori
    #Input: 
    #    Ccos, Csin coefficients for sliced Fourier series, for frequency spectrum Omegacutfft. (nxd arrays)
    #    Testing data: test_inputs, will only be evaluated at indsIII (otherwise 0), no_testing_data: length of entire test data, no_test_samples_to_evaluate: length of test data to evaluate (acc to indsIII)
    # Output: (nxd) pred_test_outputs: evaluated fourier series at test_inputs
    batch_size = 1000


    pred_test_outputs = np.zeros((no_testing_data,Ccos.shape[1])).astype('float64') ###<----------!!!
    no_batches_test = int(np.floor(no_test_samples_to_evaluate/ batch_size)) ###<----------!!!
    for kk in range(0, no_batches_test-1):
        III_to_eval = indsIII[kk*batch_size:(kk+1)*batch_size] ###<----------!!!
        ins_batch = test_inputs[III_to_eval, :] ###<----------!!!
        outs_batch= eval_cs_real(ins_batch,Ccos,Csin,Omegacutfft) 
        pred_test_outputs[III_to_eval,:] = np.copy(outs_batch)
     
    III_to_eval = indsIII[(no_batches_test-1)*batch_size:] ###<----------!!!
    ins_batch = test_inputs[III_to_eval, :] ###<----------!!!
    outs_batch= eval_cs_real(ins_batch,Ccos,Csin,Omegacutfft)
    pred_test_outputs[III_to_eval,:] = np.copy(outs_batch)
    
    return pred_test_outputs


