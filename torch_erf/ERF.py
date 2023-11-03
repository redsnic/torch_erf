import torch

def polyval(x,coeffs):  
    """Implementation of the Horner scheme to evaluate a polynomial

    taken from https://discuss.pytorch.org/t/polynomial-evaluation-by-horner-rule/67124
    
    Args:
        x (torch.Tensor): variable 
        coeffs (torch.Tensor): coefficients of the polynomial
    """
    curVal=0
    for curValIndex in range(len(coeffs)-1):
        curVal=(curVal+coeffs[curValIndex])*x[0]
    return(curVal+coeffs[len(coeffs)-1])


class ERF_1994(torch.nn.Module): 
    """Class to compute the error function of a complex number (extends torch.special.erf behavior)

    This class is based on the algorithm proposed in:
    Weideman, J. Andre C. "Computation of the complex error function." SIAM Journal on Numerical Analysis 31.5 (1994): 1497-1518
    """
    def __init__(self, n_coefs):
        """Defaul constructor

        Args:
            n_coefs (integer): The number of polynomial coefficients to use in the approximation
        """
        super(ERF_1994, self).__init__()
        # compute polynomial coefficients and other constants
        self.N = n_coefs
        self.i = torch.complex(torch.tensor(0.),torch.tensor(1.))
        self.M = 2*self.N
        self.M2 = 2*self.M
        self.k = torch.linspace(-self.M+1, self.M-1, self.M2-1)
        self.L = torch.sqrt(self.N/torch.sqrt(torch.tensor(2.)))
        self.theta = self.k*torch.pi/self.M
        self.t = self.L*torch.tan(self.theta/2)
        self.f = torch.exp(-self.t**2)*(self.L**2 + self.t**2)
        self.a = torch.fft.fft(torch.fft.fftshift(self.f)).real/self.M2
        self.a = torch.flipud(self.a[1:self.N+1])

    def w_algorithm(self, z):
        """Compute the Faddeeva function of a complex number

        The constant coefficients are computed in the constructor of the class.

        Weideman, J. Andre C. "Computation of the complex error function." SIAM Journal on Numerical Analysis 31.5 (1994): 1497-1518

        Args:
            z (torch.Tensor): A tensor of complex numbers (any shape is allowed)

        Returns:
            torch.Tensor: w(z) for each element of z
        """
        Z = (self.L+self.i*z)/(self.L-self.i*z)
        p = polyval(Z.unsqueeze(0), self.a)
        w = 2*p/(self.L-self.i*z)**2+(1/torch.sqrt(torch.tensor(torch.pi)))/(self.L-self.i*z)
        return w
    
    def forward(self, z):
        """Compute the error function of a complex number

        The result is computed by manipulating the Faddeeva function.

        Args:
            z (torch.Tensor): A tensor of complex numbers (any shape is allowed)

        Returns:
            torch.Tensor: erf(z) for each element of z
        """
        # exploit the symmetry of the error function
        # find the sign of the real part
        sign_r = torch.sign(z.real)
        sign_i = torch.sign(z.imag)
        # flip sign of imaginary part if negative
        z = torch.complex(torch.abs(z.real), torch.abs(z.imag))
        out = -torch.exp(torch.log(self.w_algorithm(z*self.i)) - z**2) + 1
        return torch.complex(out.real*sign_r, out.imag*sign_i)

    def backward(self, z):
        """Compute the gradient of the error function of a complex number.

        As we know the analytical derivative of the the error function, we can use it directly.
        
        Args:
            z (torch.Tensor): A tensor of complex numbers (any shape is allowed)
        Returns:
            torch.Tensor: grad(erf(z)) for each element of x
        """
        return 2/torch.sqrt(torch.tensor(torch.pi))*torch.exp(-z**2)
    
