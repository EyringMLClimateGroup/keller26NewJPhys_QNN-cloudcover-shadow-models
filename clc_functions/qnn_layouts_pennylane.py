import math
import pennylane as qml
import pennylane.numpy as np




### This function returns the number of trainable gate angles for a given
### circuit architecture defined in this file, specified by a string
### 'name_arch'
def no_of_angles_pqc(name_arch, no_qubits, n_enc, n_dec):
    no_gate_angles = 0

    if name_arch=='CNOTPBC':
        no_gate_angles = 2*n_enc*no_qubits + 3*n_dec*no_qubits
    if name_arch=='CNOTNN':
        no_gate_angles = 2*n_enc*no_qubits + 3*n_dec*no_qubits
    if name_arch=='XYZ':
        no_gate_angles = 3*n_enc*(no_qubits-1) + 3*n_dec*(no_qubits-1) + n_dec*no_qubits
    if name_arch=='ZZXY':
        no_gate_angles = n_enc*(no_qubits-1) + n_enc*no_qubits + n_dec*(no_qubits-1) + 2*n_dec*no_qubits
    if name_arch=='IONS':
        no_gate_angles = n_enc*(no_qubits+1) + 3*n_dec*(no_qubits+1)
    if name_arch=='CNOTPBC_HONNenc':
        no_gate_angles = 2*n_enc*no_qubits + 3*n_dec*no_qubits
    if name_arch=='CNOTNN_HONNenc':
        no_gate_angles = 2*n_enc*no_qubits + 3*n_dec*no_qubits
    if name_arch=='CNOTPBCsimple':
        no_gate_angles = n_enc*no_qubits + 2*n_dec*no_qubits
    if name_arch=='AbbasIBM':
        no_gate_angles = 2*n_dec*no_qubits
    if name_arch=='Abbas':
        no_gate_angles = 2*n_dec*no_qubits

    return no_gate_angles





########################################################################
### ------------------------- CNOT-PBC QNN ------------------------- ###
########################################################################
def CNOTPBC_circuit(data, pars, n_enc, n_dec, wires):
    """
    Generate circuit ansatz with input state: |0>,
    repeated reupload blocks as:
             _______
    -- x0 --|       |-- Ry - Rz -
            |       |     
    -- x1 --| CNOTs |-- Ry - Rz -
            |       |              for encoding,
    -- x2 --| (PBC) |-- Ry - Rz -
            |       |           
    -- x3 --|_______|-- Ry - Rz -
    
    and:
                      _______       
                   --|       |- Ry - Rz - Rx -- 
                     |       | 
                   --| CNOTs |- Ry - Rz - Rx -- 
    repeated:        |       |                     for decoding.
                   --| (PBC) |- Ry - Rz - Rx -- 
                     |       | 
                   --|_______|- Ry - Rz - Rx -- 
    
    """
    
    nb = 0
    ne = 0
    n_wires = len(wires)
    for nr in range(0,n_enc):
        # AngleEmbedding supports batching, i.e., if provided with a
        # (batch_size, n_wires) array it recognizes that it should
        # create a batch of batch_size computations
        qml.AngleEmbedding(data, wires=range(n_wires), rotation='X')
        qml.broadcast(qml.CNOT, wires=wires, pattern="ring")
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RY(th[nq], wires=nq)
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RZ(th[nq], wires=nq)

    for nr in range(0,n_dec):
        qml.broadcast(qml.CNOT, wires=wires, pattern="ring")
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RY(th[nq], wires=nq)
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RZ(th[nq], wires=nq)
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RX(th[nq], wires=nq)





########################################################################
### ------------------------- CNOT-NN QNN -------------------------- ###
########################################################################
def CNOTNN_circuit(data, pars, n_enc, n_dec, wires):
    """
    Generate circuit ansatz with input state: |0>,
    repeated reupload blocks as:
             _______
    -- x0 --|       |-- Ry - Rz -
            |       |     
    -- x1 --| CNOTs |-- Ry - Rz -
            |       |              for encoding,
    -- x2 --| (NNs) |-- Ry - Rz -
            |       |           
    -- x3 --|_______|-- Ry - Rz -
    
    and:
                      _______       
                   --|       |- Ry - Rz - Rx -- 
                     |       | 
                   --| CNOTs |- Ry - Rz - Rx -- 
    repeated:        |       |                     for decoding.
                   --| (NNs) |- Ry - Rz - Rx -- 
                     |       | 
                   --|_______|- Ry - Rz - Rx -- 
    
    """
    
    nb = 0
    ne = 0
    n_wires = len(wires)
    for nr in range(0,n_enc):
        # AngleEmbedding supports batching, i.e., if provided with a
        # (batch_size, n_wires) array it recognizes that it should
        # create a batch of batch_size computations
        qml.AngleEmbedding(data, wires=range(n_wires), rotation='X')
        qml.broadcast(qml.CNOT, wires=wires, pattern="chain")
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RY(th[nq], wires=nq)
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RZ(th[nq], wires=nq)

    for nr in range(0,n_dec):
        qml.broadcast(qml.CNOT, wires=wires, pattern="chain")
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RY(th[nq], wires=nq)
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RZ(th[nq], wires=nq)
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RX(th[nq], wires=nq)





########################################################################
### -------------------- simplified CNOT-PBC QNN ------------------- ###
########################################################################
def CNOTPBC_simple_circuit(data, pars, n_enc, n_dec, wires):
    """
    Generate circuit ansatz with input state: |0>,
    repeated reupload blocks as:
             _______
    -- x0 --|       |-- Ry --
            |       |     
    -- x1 --| CNOTs |-- Ry --
            |       |           for encoding,
    -- x2 --| (PBC) |-- Ry --
            |       |           
    -- x3 --|_______|-- Ry --
    
    and:
                            _______       
                   -- Rz --|       |-- Ry -- 
                           |       | 
                   -- Rz --| CNOTs |-- Ry -- 
    repeated:              |       |           for decoding.
                   -- Rz --| (PBC) |-- Ry -- 
                           |       | 
                   -- Rz --|_______|-- Ry -- 
    
    """
    
    nb = 0
    ne = 0
    n_wires = len(wires)
    for nr in range(0,n_enc):
        # AngleEmbedding supports batching, i.e., if provided with a
        # (batch_size, n_wires) array it recognizes that it should
        # create a batch of batch_size computations
        qml.AngleEmbedding(data, wires=range(n_wires), rotation='X')
        qml.broadcast(qml.CNOT, wires=wires, pattern="ring")
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RY(th[nq], wires=nq)

    for nr in range(0,n_dec):
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RZ(th[nq], wires=nq)
        qml.broadcast(qml.CNOT, wires=wires, pattern="ring")
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RY(th[nq], wires=nq)





########################################################################
### ------------ XYZ-like QNN (with indep. bond weigths) ----------- ###
########################################################################
def XYZ_circuit(data, pars, n_enc, n_dec, wires):
    """
    Generate circuit ansatz with reupload blocks as:
    
    -- x0 - Z ------ X ------ Y ------
            |        |        |       
    -- x1 - Z - Z -- X - X -- Y - Y --
                |        |        |       for encoding,
    -- x2 - Z - Z -- X - X -- Y - Y --
            |        |        |       
    -- x3 - Z ------ X ------ Y ------
    
    and:
    
                   --- Z ------ X ------ Y ------ Rx --
                       |        |        |       
                   --- Z - Z -- X - X -- Y - Y -- Rx --
    repeated:              |        |        |              for decoding
                   --- Z - Z -- X - X -- Y - Y -- Rx --
                       |        |        |       
                   --- Z ------ X ------ Y ------ Rx --
    
    """
    
    nb = 0
    ne = 0
    n_wires = len(wires)
    n_half = int(math.floor(n_wires / 2))
    nm1_half = int(math.floor((n_wires - 1) / 2))
    for nr in range(0,n_enc):
        # AngleEmbedding supports batching, i.e., if provided with a
        # (batch_size, n_wires) array it recognizes that it should
        # create a batch of batch_size computations
        qml.AngleEmbedding(data, wires=range(n_wires), rotation='X')
        nb = ne;  ne = nb + n_half;   th = pars[nb:ne];  qml.broadcast(qml.IsingZZ, wires=wires, pattern="double", parameters=th)
        nb = ne;  ne = nb + nm1_half; th = pars[nb:ne];  qml.broadcast(qml.IsingZZ, wires=wires, pattern="double_odd", parameters=th)
        nb = ne;  ne = nb + n_half;   th = pars[nb:ne];  qml.broadcast(qml.IsingXX, wires=wires, pattern="double", parameters=th)
        nb = ne;  ne = nb + nm1_half; th = pars[nb:ne];  qml.broadcast(qml.IsingXX, wires=wires, pattern="double_odd", parameters=th)
        nb = ne;  ne = nb + n_half;   th = pars[nb:ne];  qml.broadcast(qml.IsingYY, wires=wires, pattern="double", parameters=th)
        nb = ne;  ne = nb + nm1_half; th = pars[nb:ne];  qml.broadcast(qml.IsingYY, wires=wires, pattern="double_odd", parameters=th)

    for nr in range(0,n_dec):
        nb = ne;  ne = nb + n_half;   th = pars[nb:ne];  qml.broadcast(qml.IsingZZ, wires=wires, pattern="double", parameters=th)
        nb = ne;  ne = nb + nm1_half; th = pars[nb:ne];  qml.broadcast(qml.IsingZZ, wires=wires, pattern="double_odd", parameters=th)
        nb = ne;  ne = nb + n_half;   th = pars[nb:ne];  qml.broadcast(qml.IsingXX, wires=wires, pattern="double", parameters=th)
        nb = ne;  ne = nb + nm1_half; th = pars[nb:ne];  qml.broadcast(qml.IsingXX, wires=wires, pattern="double_odd", parameters=th)
        nb = ne;  ne = nb + n_half;   th = pars[nb:ne];  qml.broadcast(qml.IsingYY, wires=wires, pattern="double", parameters=th)
        nb = ne;  ne = nb + nm1_half; th = pars[nb:ne];  qml.broadcast(qml.IsingYY, wires=wires, pattern="double_odd", parameters=th)
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RX(th[nq], wires=nq)





########################################################################
### ------------- ZZ-X-Y QNN (with indep. bond weigths) ------------ ###
########################################################################
def ZZXY_circuit(data, pars, n_enc, n_dec, wires):
    """
    Generate circuit ansatz with reupload blocks as:
    
    -- x0 -- Z ------ Ry --
             |                  
    -- x1 -- Z - Z -- Ry --
                 |             for encoding,
    -- x2 -- Z - Z -- Ry --
             |               
    -- x3 -- Z ------ Ry --
    
    and:
    
                   -- Rx -- Z ------ Ry --
                            |           
                   -- Rx -- Z - Z -- Ry --
    repeated:                   |              for decoding
                   -- Rx -- Z - Z -- Ry --
                            |        
                   -- Rx -- Z ------ Ry --
    
    """
    
    nb = 0
    ne = 0
    n_wires = len(wires)
    n_half = int(math.floor(n_wires / 2))
    nm1_half = int(math.floor((n_wires - 1) / 2))
    for nr in range(0,n_enc):
        # AngleEmbedding supports batching, i.e., if provided with a
        # (batch_size, n_wires) array it recognizes that it should
        # create a batch of batch_size computations
        qml.AngleEmbedding(data, wires=range(n_wires), rotation='X')
        nb = ne;  ne = nb + n_half;   th = pars[nb:ne];  qml.broadcast(qml.IsingZZ, wires=wires, pattern="double", parameters=th)
        nb = ne;  ne = nb + nm1_half; th = pars[nb:ne];  qml.broadcast(qml.IsingZZ, wires=wires, pattern="double_odd", parameters=th)
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RY(th[nq], wires=nq)

    for nr in range(0,n_dec):
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RX(th[nq], wires=nq)
        nb = ne;  ne = nb + n_half;   th = pars[nb:ne];  qml.broadcast(qml.IsingZZ, wires=wires, pattern="double", parameters=th)
        nb = ne;  ne = nb + nm1_half; th = pars[nb:ne];  qml.broadcast(qml.IsingZZ, wires=wires, pattern="double_odd", parameters=th)
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RY(th[nq], wires=nq)





########################################################################
### ---------------------- TrappedIon-like QNN --------------------- ###
########################################################################
def TrappedIonLike_circuit(data, pars, n_enc, n_dec, wires):
    """
    Generate circuit ansatz with input state: Prod_j H(j)|0>,
    reupload blocks as (XX: trapped-ion-like XX interaction from COM coupling):
             ____
    -- z0 --|    |-- Ry --
            |    | 
    -- z1 --|    |-- Ry --
            | XX |          for encoding,
    -- z2 --|    |-- Ry --
            |    |       
    -- z3 --|____|-- Ry --
    
    and:
                                 ____       
                   -- Rx -- Rz -|    |- Ry --
                                |    |      
                   -- Rx -- Rz -|    |- Ry --
    repeated:                   | XX |         for decoding
                   -- Rx -- Rz -|    |- Ry --
                                |    |      
                   -- Rx -- Rz -|____|- Ry --
    
    """

    def IonTrap_XX_HamiltonianEvo(par, wires):
        """
        Implements the operator exp(-i * par * H) with:
                   H = 0.5 * \sum_{i<j} X(i)X(j) / |i-j|
        """
        n_wires = len(wires)
        for i in range(0, n_wires-1):
            for j in range(i+1, n_wires):
                Jij = 1.0 / (j - i)
                phi = par * Jij
                qml.IsingXX(phi, wires=wires.subset([i, j]))

    nb = 0
    ne = 0
    n_wires = len(wires)
    qml.broadcast(qml.Hadamard, wires=wires, pattern="single")
    for nr in range(0,n_enc):
        # AngleEmbedding supports batching, i.e., if provided with a
        # (batch_size, n_wires) array it recognizes that it should
        # create a batch of batch_size computations
        qml.AngleEmbedding(data, wires=range(n_wires), rotation='Z')
        nb = ne;  ne = nb + 1;  th = pars[nb];  IonTrap_XX_HamiltonianEvo(th, wires)
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RY(th[nq], wires=nq)

    for nr in range(0,n_dec):
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RX(th[nq], wires=nq)
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RZ(th[nq], wires=nq)
        nb = ne;  ne = nb + 1;  th = pars[nb];  IonTrap_XX_HamiltonianEvo(th, wires)
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RY(th[nq], wires=nq)





########################################################################
### ------------------------- CNOT-PBC QNN ------------------------- ###
########################################################################
def CNOTPBC_HONNenc_circuit(data, pars, n_enc, n_dec, wires):
    """
    Generate circuit ansatz with input state: |0>,
    repeated reupload blocks as:
                                  _______
    -- x0 -- xx0 ----------------|       |-- Ry - Rz -
              |                  |       |     
    -- x1 -- xx1 -- xx1 ---------| CNOTs |-- Ry - Rz -
                     |           |       |              for encoding,
    -- x2 --------- xx2 -- xx2 --| (PBC) |-- Ry - Rz -
                            |    |       |           
    -- x3 ---------------- xx3 --|_______|-- Ry - Rz -
    
    and:
                      _______       
                   --|       |- Ry - Rz - Rx -- 
                     |       | 
                   --| CNOTs |- Ry - Rz - Rx -- 
    repeated:        |       |                     for decoding.
                   --| (PBC) |- Ry - Rz - Rx -- 
                     |       | 
                   --|_______|- Ry - Rz - Rx -- 
    
    """
    
    nb = 0
    ne = 0
    n_wires = len(wires)
    dataT = data.transpose()
    for nr in range(0,n_enc):
        # AngleEmbedding supports batching, i.e., if provided with a
        # (batch_size, n_wires) array it recognizes that it should
        # create a batch of batch_size computations
        qml.AngleEmbedding(data, wires=range(n_wires), rotation='X')

        # The parametrized qml gates support vectorization, i.e., if 
        # provided with an array (batch_size, ) they recognize that
        # they should create a batch of batch_size computations
        for nq in range(0,n_wires-1):
            qml.IsingXX(dataT[nq]*dataT[nq+1]/(2.0*np.pi), wires=wires.subset([nq, nq+1]))
        
        qml.broadcast(qml.CNOT, wires=wires, pattern="ring")
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RY(th[nq], wires=nq)
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RZ(th[nq], wires=nq)

    for nr in range(0,n_dec):
        qml.broadcast(qml.CNOT, wires=wires, pattern="ring")
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RY(th[nq], wires=nq)
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RZ(th[nq], wires=nq)
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RX(th[nq], wires=nq)





########################################################################
### ------------------------- CNOT-NN QNN -------------------------- ###
########################################################################
def CNOTNN_HONNenc_circuit(data, pars, n_enc, n_dec, wires):
    """
    Generate circuit ansatz with input state: |0>,
    repeated reupload blocks as:
                                  _______
    -- x0 -- xx0 ----------------|       |-- Ry - Rz -
              |                  |       |     
    -- x1 -- xx1 -- xx1 ---------| CNOTs |-- Ry - Rz -
                     |           |       |              for encoding,
    -- x2 --------- xx2 -- xx2 --| (NNs) |-- Ry - Rz -
                            |    |       |           
    -- x3 ---------------- xx3 --|_______|-- Ry - Rz -
    
    and:
                      _______       
                   --|       |- Ry - Rz - Rx -- 
                     |       | 
                   --| CNOTs |- Ry - Rz - Rx -- 
    repeated:        |       |                     for decoding.
                   --| (NNs) |- Ry - Rz - Rx -- 
                     |       | 
                   --|_______|- Ry - Rz - Rx -- 
    
    """
    
    nb = 0
    ne = 0
    n_wires = len(wires)
    dataT = data.transpose()
    for nr in range(0,n_enc):
        # AngleEmbedding supports batching, i.e., if provided with a
        # (batch_size, n_wires) array it recognizes that it should
        # create a batch of batch_size computations
        qml.AngleEmbedding(data, wires=range(n_wires), rotation='X')

        # The parametrized qml gates support vectorization, i.e., if 
        # provided with an array (batch_size, ) they recognize that
        # they should create a batch of batch_size computations
        for nq in range(0,n_wires-1):
            qml.IsingXX(dataT[nq]*dataT[nq+1]/(2.0*np.pi), wires=wires.subset([nq, nq+1]))
        
        qml.broadcast(qml.CNOT, wires=wires, pattern="chain")
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RY(th[nq], wires=nq)
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RZ(th[nq], wires=nq)

    for nr in range(0,n_dec):
        qml.broadcast(qml.CNOT, wires=wires, pattern="chain")
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RY(th[nq], wires=nq)
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RZ(th[nq], wires=nq)
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RX(th[nq], wires=nq)





########################################################################
### ----------------- Abbas et al. QNN (IBM device) ---------------- ###
########################################################################
def AbbasIBM_circuit(data, pars, n_enc, n_dec, wires):
    """
    Generate circuit ansatz with input state: |0>,
    reupload blocks as:
    
    -- H -- z0 - zz0 --------------
                  |          
    -- H -- z1 - zz1 - zz1 --------
                        |            for encoding,
    -- H -- z2 ------- zz2 - zz2 --
                              |            
    -- H -- z3 ------------- zz3 --
    
    and:
    
                   -- Ry -- . ------------ Ry --
                            |        
                   -- Ry -- X -- . ------- Ry --
    repeated:                    |                 for decoding
                   -- Ry ------- X -- . -- Ry --
                                      |       
                   -- Ry ------------ X -- Ry --
    
    """

    nb = 0
    ne = 0
    n_wires = len(wires)
    dataT = data.transpose()
    
    for nr in range(0,n_enc):
        qml.broadcast(qml.Hadamard, wires=wires, pattern="single")
        
        # AngleEmbedding supports batching, i.e., if provided with a
        # (batch_size, n_wires) array it recognizes that it should
        # create a batch of batch_size computations
        qml.AngleEmbedding(data, wires=range(n_wires), rotation='Z')

        # The parametrized qml gates support vectorization, i.e., if 
        # provided with an array (batch_size, ) they recognize that
        # they should create a batch of batch_size computations
        for nq in range(0,n_wires-1):
            qml.IsingZZ(dataT[nq]*dataT[nq+1]/(2.0*np.pi), wires=wires.subset([nq, nq+1]))

    for nr in range(0,n_dec):
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RY(th[nq], wires=nq)
        qml.broadcast(qml.CNOT, wires=wires, pattern="chain")
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RY(th[nq], wires=nq)





########################################################################
### -------------- Abbas et al. QNN (hard feature map) ------------- ###
########################################################################
def Abbas_circuit(data, pars, n_enc, n_dec, wires):
    """
    Generate circuit ansatz with input state: |0>,
    reupload blocks as:
    
    -- H -- z0 - zz0 ------------- zz0 - zz0 ----
                  |                 |     |  
    -- H -- z1 - zz1 - zz1 - zz1 ---|-----|------
                        |     |     |     |         for encoding,
    -- H -- z2 - zz2 - zz2 ---|--- zz2 ---|------
                  |           |           |       
    -- H -- z3 - zz3 ------- zz3 ------- zz3 ----
    
    and:
    
                   -- Ry - . --------- . - . -- Ry --
                           |           |   |   
                   -- Ry - X - . - . --| - | -- Ry --
    repeated:                  |   |   |   |           for decoding
                   -- Ry - . - X --|-- X --|--- Ry --
                           |       |       |
                   -- Ry - X ----- X ----- X -- Ry --
    
    """

    nb = 0
    ne = 0
    n_wires = len(wires)
    dataT = data.transpose()
    
    for nr in range(0,n_enc):
        qml.broadcast(qml.Hadamard, wires=wires, pattern="single")
        
        # AngleEmbedding supports batching, i.e., if provided with a
        # (batch_size, n_wires) array it recognizes that it should
        # create a batch of batch_size computations
        qml.AngleEmbedding(data, wires=range(n_wires), rotation='Z')

        # The parametrized qml gates support vectorization, i.e., if 
        # provided with an array (batch_size, ) they recognize that
        # they should create a batch of batch_size computations
        for nq1 in range(0,n_wires-1):
            for nq2 in range(nq1+1,n_wires):
                qml.IsingZZ(dataT[nq1]*dataT[nq2]/(2.0*np.pi), wires=wires.subset([nq1, nq2]))

    for nr in range(0,n_dec):
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RY(th[nq], wires=nq)
        qml.broadcast(qml.CNOT, wires=wires, pattern="all_to_all")
        nb = ne
        ne = nb + n_wires
        th = pars[nb:ne]
        for nq in range(0,n_wires):
            qml.RY(th[nq], wires=nq)
