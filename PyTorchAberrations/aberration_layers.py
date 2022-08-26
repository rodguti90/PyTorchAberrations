import torch
from torch.nn import Module



################################################################
###################### AUX FUNCTIONS ###########################
################################################################


def zernike_Z(j, X, Y):

    # see https://en.wikipedia.org/wiki/Zernike_polynomials
    if j == 0:
        F = torch.ones_like(X)
    elif j == 1:
        F = 2*X
    elif j == 2:
        F = 2*Y
    elif j == 3:
        # Oblique astigmatism
        F = 2.*(6.**(1/2))*X.mul(Y)
    elif j == 4:
        # Defocus
        F = (3.**(1/2))*(2.*(X**2+Y**2)-1)
    elif j == 5:
        # Vertical astigmatism
        F = (6.**(1/2))*(X**2-Y**2)
    else:
        R = torch.sqrt(X**2+Y**2)
        THETA = torch.atan2(Y, X)
        if j == 6:
            # Vertical trefoil 
            F = (8.**(1/2))*torch.mul(R**3, torch.sin(3.*THETA))
        elif j == 7:
            # Vertical coma
            F = (8.**(1/2))*torch.mul(3.*R**3-2.*R,torch.sin(3.*THETA))
        elif j == 8:
            # Horizontal coma 
            F = (8.**(1/2))*torch.mul(3.*R**3-2.*R,torch.cos(3.*THETA))
        elif j == 9:
            # Oblique trefoil 
            F = (8.**(1/2))*torch.mul(R**3, torch.cos(3.*THETA))
        elif j == 10:
            # Oblique quadrafoil 
            F = (10.**(1/2))*torch.mul(R**4, torch.sin(4.*THETA))
        elif j == 11:
            # Oblique secondary astigmatism 
            F = (10.**(1/2))*torch.mul(4.*R**4-3.*R**2, torch.sin(2.*THETA))
        elif j == 12:
            # Primary spherical
            F = (5.**(1/2))*(6.*R**4-6.*R**2 + torch.ones_like(R))
        elif j == 13:
            # Vertical secondary astigmatism 
            F = (10.**(1/2))*torch.mul(4.*R**4-3.*R**2, torch.cos(2.*THETA))
        elif j == 14:
            # Vertical quadrafoil 
            F = (10.**(1/2))*torch.mul(R**4, torch.cos(4.*THETA))
        else:
            raise
    
    return F

#######################################################
#################### MODULES ##########################
#######################################################

# class ComplexZeroPad2d(Module):
#     '''
#     Apply zero padding to a batch of 2D complex images (or matrix)
#     '''
#     def __init__(self, padding):
#         super(ComplexZeroPad2d, self).__init__()
#         self.pad_r = ZeroPad2d(padding)
#         self.pad_i = ZeroPad2d(padding)

#     def forward(self,input):
#         return torch.stack((self.pad_r(input[...,0]), 
#                            self.pad_i(input[...,1])), dim = -1)     

class ComplexZernike(Module):
    '''
    Layer that apply a complex Zernike polynomial to the phase of a batch 
    of compleximages (or a matrix).
    Only one parameter, the strenght of the polynomial, is learned.
    Initial value is 0.
    '''
    def __init__(self, j):
        super(ComplexZernike, self).__init__()
        assert j in range(15)
        self.j = j
        self.alpha = torch.nn.Parameter(torch.zeros(1), requires_grad=True)


    def forward(self, input):

        nx = torch.arange(0,2,2./input.shape[1], dtype = torch.float32)
        ny = torch.arange(0,2,2./input.shape[2], dtype = torch.float32)

        X0, Y0 = 1.+1./input.shape[1], 1.+1./input.shape[2]
        X,Y = torch.meshgrid(nx,ny)
        X = X.to(input.device)-X0
        Y = Y.to(input.device)-Y0

        F = zernike_Z(self.j, X, Y)

        return input * torch.exp(1j*self.alpha*F)

class ComplexScaling(Module):
    '''
    Layer that apply a global scaling to a stack of 2D complex images (or matrix).
    Only one parameter, the scaling factor, is learned. 
    Initial value is 1.
    '''
    def __init__(self):
        super(ComplexScaling, self).__init__()
        
        self.theta = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        # parameters 0 and 4 are the ones corresponding to x and y scaling
        # parameters 1 and 3 are the ones corresponding to shearing
        # parameters 2 and 6 are shifts

    def forward(self, input):
            input = torch.view_as_real(input).permute((0,3,1,2))

            grid = torch.nn.functional.affine_grid(
                ((1.+self.theta)*(torch.tensor([1, 0., 0., 0., 1, 0.],
                                         dtype=input.dtype).to(input.device))
                ).reshape((2,3)).expand((input.shape[0],2,3)), 
                                 input.size())                      
                                         
            return torch.view_as_complex(torch.nn.functional.grid_sample(input, grid, align_corners=True).permute((0,2,3,1)).contiguous())
        
class ComplexDeformation(Module):
    '''
    Layer that apply a global affine transformation to a stack of 2D complex images (or matrix).
    6 parameters are learned.
    '''
    def __init__(self):
        super(ComplexDeformation, self).__init__()
        
        self.theta = torch.nn.Parameter(torch.tensor([0., 0, 0, 0, 0., 0]))
        # parameters 0 and 4 are the ones corresponding to x and y scaling
        # parameters 1 and 3 are the ones corresponding to shearing
        # parameters 2 and 6 are shifts

    def forward(self, input):
            input = torch.view_as_real(input).permute((0,3,1,2))
            grid = torch.nn.functional.affine_grid(
                ((1.+self.theta).mul(torch.tensor([1, 0., 0., 0., 1, 0.],
                                         dtype=input.dtype).to(input.device))
                ).reshape((2,3)).expand((input.shape[0],2,3)), 
                                 input.size())                 

            return torch.view_as_complex(torch.nn.functional.grid_sample(input, grid, align_corners=True).permute((0,2,3,1)))

