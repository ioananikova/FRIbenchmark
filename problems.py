import enum
import torch
import math
    
def donut(x):
    v = torch.exp(-2 * (x[:, 0] - 0.5) ** 2 - 4 * (x[:, 1] - 0.6) ** 2)
    return torch.stack((v, v), dim=-1)

def sinexp(x):
    result = x[:, 0] * torch.sin(5 * torch.pi * x[:, 1]) + torch.exp(x[:, 1] - 0.5) - 0.5
    return result.unsqueeze(-1)

def branin(x):
    a = 1.0
    b = 5.1 / (4 * torch.pi ** 2)
    c = 5 / torch.pi
    r = 6.0
    s = 10.0
    t = 1 / (8 * torch.pi)
    
    term1 = a * (x[:, 1] - b * x[:, 0] ** 2 + c * x[:, 0] - r) ** 2
    term2 = s * (1 - t) * torch.cos(x[:, 0]) + s
    return (term1 + term2).unsqueeze(-1)

def himmelblau(x):
    term1 = (x[:, 0] ** 2 + x[:, 1] - 11) ** 2
    term2 = (x[:, 0] + x[:, 1] ** 2 - 7) ** 2
    return (term1 + term2).unsqueeze(-1)

def hosaki(x):
    term1 = (1 - 8 * x[:, 0] + 7 * x[:, 0] ** 2 - (7 / 3) * x[:, 0] ** 3 + (1 / 4) * x[:, 0] ** 4)
    term2 = x[:, 1] ** 2 * torch.exp(-x[:, 1])
    return (term1 * term2).unsqueeze(-1)

def cheese(x):
    f1 = x[:,1] - 2.0*x[:,0] - 15.0
    f2 = x[:,0]**2/2.0 + 4.0*x[:,0] - 5.0 - x[:,1]
    f3 = 10.0 - (x[:,0] - 4.0)**2/5.0 - x[:,1]**2/0.5
    f4 = x[:,1] - 15.0
    f5 = x[:,1]*(6.0 + x[:,0]) - 80.0
    f6 = - (x[:,0] + 5)**2 - (x[:,1] + 5)**2 + 2.0
    return torch.stack((f1, f2, f3, f4, f5, f6), dim=-1)

def sasena(x):
    f1 = (x[:, 0] - 3.0) ** 2 + (x[:, 1] + 2.0) ** 2 * torch.exp(-x[:,1]**7) - 12.0
    f2 = 10.0*x[:,0] + x[:,1] - 7.0
    f3 = (x[:,0] - 0.5)**2 + (x[:,1] - 0.5)**2 - 0.2
    return torch.stack((f1, f2, f3), dim=-1)

def sixhumpcamel(x):
    term1 = (4 - 2.1 * x[:, 0] ** 2 + (x[:, 0] ** 4) / 3) * x[:, 0] ** 2
    term2 = x[:, 0] * x[:, 1]
    term3 = (-4 + 4 * x[:, 1] ** 2) * x[:, 1] ** 2
    return (term1 + term2 + term3).unsqueeze(-1)

def crescent(x):
    # in 2D:
    # g1 = 4.84 - (x[:, 0] - 0.5) ** 2 - (x[:, 1] - 2.5) ** 2
    # g2 = x[:, 0]**2 + (x[:, 1] - 2.5) ** 2 - 4.84

    c1 = torch.tensor([1.0, 2.5, 2.5, 2.5, 2.5, 2.5])
    c2 = torch.tensor([0.0, 2.5, 2.5, 2.5, 2.5, 2.5])
    R2 = 4.0

    g1 = R2 - torch.sum((x - c1) ** 2, dim=1)
    g2 = torch.sum((x - c2) ** 2, dim=1) - R2
    return torch.stack((g1, g2), dim=-1)

def uneq_spheres(x):
    centers = torch.tensor([[0.2, 0.2, 0.2], [0.8, 0.2, 0.5], [0.8, 0.8, 0.8], [0.2, 0.8, 0.5], [0.5, 0.5, 0.5]])
    radii = torch.tensor([0.05, 0.1, 0.2, 0.05, 0.1])
    distances = torch.sqrt(torch.sum((x.unsqueeze(1) - centers.unsqueeze(0)) ** 2, dim=2))
    min_distances = distances - radii.unsqueeze(0)
    min_distance, _ = torch.min(min_distances, dim=1)
    return min_distance.unsqueeze(-1)

def eq_spheres(x):
    centers = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2], [0.8, 0.2, 0.5, 0.5, 0.2], [0.8, 0.8, 0.8, 0.8, 0.8], [0.2, 0.8, 0.5, 0.5, 0.8], [0.5, 0.5, 0.5, 0.5, 0.5]])
    radii = torch.tensor([0.18, 0.18, 0.18, 0.18, 0.18])
    distances = torch.sqrt(torch.sum((x.unsqueeze(1) - centers.unsqueeze(0)) ** 2, dim=2))
    min_distances = distances - radii.unsqueeze(0)
    min_distance, _ = torch.min(min_distances, dim=1)
    return min_distance.unsqueeze(-1)

def nowackibeam(x):
    L = 0.5 # 500 #0.5 # length in m
    load = 5e3 # Force in N
    sigma_y = 240e6 #240 #240e6 # yield stress in Pa
    E = 216.62e9 #216620 #216.62e9 # Young's modulus in Pa
    nu = 0.27 # Poisson's ratio
    G = 86.65e9 #86650 #86.65e9 # shear modulus in Pa
    factor = 2 # safety factor

    b,h = x[:,0], x[:,1]
    f1 = h*b - 0.0025 #2500
    I_y = b*h**3/12
    f2 = load * L**3 / (3*E*I_y) - 0.005 #5
    f3 = 6 * load * L / (b * h**2) - sigma_y
    f4 = 1.5 * load / (h * b) - sigma_y/2
    I_z = b**3 * h / 12
    I_t = I_y + I_z
    f5 = load * factor - (4/L**2 * torch.sqrt(G * I_t * E * I_z / (1 - nu**2)))
    f6 = h/b - 10.0

    return torch.stack((f1, f2, f3, f4, f5, f6), dim=-1)

def tensionspring(x):
    f1 = (4 * x[:, 1]**2 - x[:, 0] * x[:, 1]) / (12566 * (x[:, 1] * x[:, 0]**3 - x[:, 0]**4)) \
            + 1 / (5108 * x[:, 0]**2) - 1
    f2 = 1 - (140.45 * x[:, 0]) / (x[:, 1]**2 * x[:, 2])
    f3 = (x[:, 0] + x[:, 1]) / 1.5 - 1
    return torch.stack((f1, f2, f3), dim=-1)

def speedreducer(x):

    x1, x2, x3, x4, x5, x6, x7 = x[:,0], x[:,1], x[:,2], x[:,3], x[:,4], x[:,5], x[:,6]
    f1 = 27 - x1*x2**2*x3 # OG 1
    f2 = 1.93 - (x2*x3*x6**4)/x4**3 # OG 3
    f3 = 1.93 - (x2*x3*x7**4)/x5**3 # OG 4
    f4 = x2*x3 - 40 # OG 7
    f5 = x1/x2 - 12 # OG 9
    f6 = 1.5*x6 - x4 + 1.9 # OG 10
    f7 = 1.1*x7 - x5 + 1.9 # OG 11
    f8 = 397.5 - (x1*x2**2*x3**2)# OG 2
    f9 = 10.0/x6**3 * torch.sqrt(((745.0*x4)/(x2*x3))**2 + 1.69e7) - 1100 # OG 5
    f10 = 10.0/x7**3 * torch.sqrt(((745.0*x5)/(x2*x3))**2 + 1.575e8) - 850 # OG 6
    f11 = 5 - x1/x2 # OG 8 - most restrictive constraint

    return torch.stack((f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11), dim=-1)


class ProblemType(enum.Enum):
    """
    Enumeration of benchmark problems with associated functions, constraints, and bounds.
        Each problem is defined by:
        - problem_name: String identifier for the problem.
        - problem_function: Callable function that computes the objective and constraints.
        - constraints: List of tuples defining constraint types and thresholds.
        - bounds: List defining the lower and upper bounds for each dimension.
    """
    donut = ("donut", donut, [("lt", 0.75), ("gt", 0.55)], [[0.0, 0.0], [1.0, 1.0]])
    branin = ("branin", branin, [("lt", 8.0)], [[-13.0, -8.0], [14.0, 23.0]])
    himmelblau = ("himmelblau", himmelblau, [("lt", 20.0)], [[-5.0, -5.0], [5.0, 5.0]])
    hosaki = ("hosaki", hosaki, [("lt", -1.1)], [[0.0, 0.0], [5.0, 6.0]])
    sinexp = ("sinexp", sinexp, [("lt", 0.001)], [[0.0, 0.0], [1.0, 1.0]])
    cheese = ("cheese", cheese, 
    [("lt", 0.0), ("lt", 0.0), ("lt", 0.0), ("lt", 0.0), ("lt", 0.0), ("lt", 0.0)], [[-10.0, -15.0], [5.0, 15.0]])
    sasena = ("sasena", sasena, [("lt", 0.0), ("lt", 0.0), ("lt", 0.0)], [[0.0, 0.0], [1.0, 1.0]])
    sixhumpcamel = ("sixhumpcamel", sixhumpcamel, [("lt", 0.1)], [[-1.0, -1.0], [1.0, 1.0]])
    crescent = ("crescent", crescent, [("gt", 0.0), ("gt", 0.0)], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [6.0, 6.0, 6.0, 6.0, 6.0, 6.0]])
    uneq_spheres = ("uneq_spheres", uneq_spheres, [("lt", 0.0)], [[0.0]*3, [1.0]*3])
    eq_spheres = ("eq_spheres", eq_spheres, [("lt", 0.0)], [[0.0]*5, [1.0]*5])
    speedreducer = ("speedreducer", speedreducer, 
                    [("lt", 0.0), ("lt", 0.0), ("lt", 0.0), ("lt", 0.0), ("lt", 0.0), ("lt", 0.0), ("lt", 0.0), ("lt", 0.0), ("lt", 0.0), ("lt", 0.0), ("lt", 0.0)], 
                    [[2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0], [3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5]])
    nowackibeam = ("nowackibeam", nowackibeam, 
                   [("lt", 0.0), ("lt", 0.0), ("lt", 0.0), ("lt", 0.0), ("lt", 0.0), ("lt", 0.0)], 
                   [[0.005, 0.02], [0.1, 0.25]])
    tensionspring = ("tensionspring", tensionspring, 
                     [("lt", 0.0), ("lt", 0.0), ("lt", 0.0)],
                     [[0.05, 0.25, 2.0], [2.0, 1.3, 15.0]])

            
    def __init__(self, problem_name, problem_function, constraints, bounds):
        self.problem_name = problem_name
        self.problem_function = problem_function
        self.constraints = constraints
        self.bounds = bounds
    
    @classmethod
    def get_problem(cls, name):
        """Get problem function by string name"""
        for problem in cls:
            if problem.problem_name == name:
                return problem.problem_function
        raise ValueError(f"Problem '{name}' not found. Available problems: {[p.problem_name for p in cls]}")
    
    def __call__(self, x):
        """Allow calling the enum member directly"""
        return self.problem_function(x)
    
    def get_constraints(self):
        """Get constraints associated with the problem"""
        return self.constraints
    
    def get_bounds(self):
        """Get bounds associated with the problem"""
        return self.bounds
    
    def num_constraints(self):
        """Get number of constraints associated with the problem"""
        return len(self.constraints) if self.constraints is not None else 0
    
    def num_dimensions(self):
        """Get number of dimensions associated with the problem"""
        return len(self.bounds[0]) if self.bounds is not None else 0
    
