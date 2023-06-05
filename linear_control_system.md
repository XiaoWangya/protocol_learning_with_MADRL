# Quick Start

## 
Both `control.place()` and `solve_discrete_are()` are used in control system design to compute an optimal feedback matrix (or gain) for a given linear time-invariant (LTI) system. However, the two functions differ in their approach to solving the problem.

The `control.place()` function uses the pole placement method, which involves selecting the desired closed-loop poles of the system and computing a corresponding feedback matrix that places these poles at the desired locations. This method is straightforward and easy to implement, but it does not guarantee optimality or robustness since it only considers the desired pole locations and not other factors such as disturbance rejection or noise sensitivity.

On the other hand, `solve_discrete_are()` uses the algebraic Riccati equation (ARE) to find a stabilizing solution to the optimal control problem. This method computes the optimal feedback gain by minimizing a quadratic cost function subject to constraints on the system dynamics and control inputs. This approach takes into account both the stability and performance requirements of the system and provides a more rigorous and robust solution than the pole placement method.

Furthermore, `solve_discrete_are()` is specifically designed for discrete-time systems, whereas `control.place()` can be used for both continuous and discrete-time systems. 

In summary, while both `control.place()` and `solve_discrete_are()` are used to design an optimal feedback matrix for LTI systems, they differ in their methodology and level of optimality and robustness. The choice of function depends on the specific requirements and constraints of the system being designed.