.. module:: ecyglpki

.. testsetup:: *

    from ecyglpki import Problem

##########################
CNF Satisfiability Problem
##########################

Introduction
============

The *Satisfiability Problem (SAT)* is a classic combinatorial problem. Given a
Boolean formula of :math:`n` variables

.. math::
   :label: 1.1

   f(x_1,x_2,\dots,x_n),

this problem is to find such values of the variables, on which the formula
takes on the value *true*.

The *CNF Satisfiability Problem (CNF-SAT)* is a version of the Satisfiability
Problem, where the Boolean formula :eq:`1.1` is specified in the
*Conjunctive Normal Form (CNF)*, that means that it is a conjunction of
*clauses*, where a clause is a disjunction of *literals*, and a literal is a
variable or its negation.
For example:

.. math::
   :label: 1.2

   (x_1\vee x_2)\;\&\;(\neg x_2\vee x_3\vee\neg x_4)\;\&\;(\neg x_1\vee x_4).

Here :math:`x_1`, :math:`x_2`, :math:`x_3`, :math:`x_4` are Boolean variables
to be assigned, :math:`\neg` means negation (logical *not*), :math:`\vee` means
disjunction (logical *or*), and :math:`\&` means conjunction (logical *and*).
One may note that the formula :eq:`1.2` is *satisfiable*, because on
:math:`x_1=` *true*, :math:`x_2=` *false*, :math:`x_3=` *false*, and
:math:`x_4=` *true* it takes on the value *true*. If a formula
is not satisfiable, it is called *unsatisfiable*, that means that
it takes on the value *false* on any values of its variables.

Any CNF-SAT problem can be easily translated to a 0-1 programming problem as
follows.

A Boolean variable :math:`x` can be modeled by a binary variable in a natural
way: :math:`x=1` means that :math:`x` takes on the value *true*, and
:math:`x=0` means that :math:`x` takes on the value *false*. Then, if a literal
is a negated variable, i.e. :math:`t=\neg x`, it can be expressed as
:math:`t=1-x`. Since a formula in CNF is a conjunction of clauses, to provide
its satisfiability we should require all its clauses to  take on the value
*true*. A particular clause is a disjunction of literals:

.. math::
   :label: 1.3

   t\vee t'\vee t''\vee\dots,

so it takes on the value *true* iff at least one of its literals takes on the
value *true*, that can be expressed as the following inequality constraint:

.. math::
   :label: 1.4

   t+t'+t''+\dots\geq 1.

Note that no objective function is used in this case, because only a feasible
solution needs to be found.

For example, the formula :eq:`1.2` can be translated to the following constraints:

.. math::
   :nowrap:

   \begin{alignat*}{8}
     &&x_1& &{}+{} &&x_2& & &&& & &&& &\geq 1\\
     &&& & &(1-{}&x_2&) &{}+{} &&x_3& &{}+{} &(1-{}&x_4&) &\geq 1\\
     &(1-{}&x_1&) & &&& & &&& &{}+{} &&x_4& &\geq 1
   \end{alignat*}

with :math:`x_1, x_2, x_3, x_4\in\{0,1\}`.

Carrying out all constant terms to the right-hand side gives corresponding 0-1
programming problem in the standard format:

.. math::
   :nowrap:

   \begin{alignat*}{5}
     & &x_1 &{}+{} &x_2 & & & & &\geq &1\\
     & & &{}-{} &x_2 &{}+{} &x_3 &{}-{} &x_4 &\geq -&1\\
     &-&x_1 & & & & &{}+{} &x_4 &\geq &0\\
   \end{alignat*}

with :math:`x_1, x_2, x_3, x_4\in\{0,1\}`.

In general case translation of a CNF-SAT problem results in the following 0-1
programming problem:

.. math::
   :label: 1.5

   \sum_{j\in J^+_i}x_j-\sum_{j\in J^-_i}x_j\geq 1-|J^-_i|,\quad i=1,\dots,m,

.. math::
   :label: 1.6

   x_j\in\{0,1\},\quad j=1,\dots,n,

where :math:`n` is the number of variables, :math:`m` is the number of clauses
(inequality constraints), :math:`J^+_i\subseteq\{1,\dots,n\}` is a
subset of variables, whose literals in :math:`i`-th clause do not have
negation, and :math:`J^-_i\subseteq\{1,\dots,n\}` is a subset of variables,
whose literals in :math:`i`-th clause are negations of that variables. It is
assumed that :math:`J^+_i\cap J^-_i=\varnothing` for all :math:`i`.


DIMACS CNF-SAT Format
=====================

.. note::

   This material is based on the paper `“Satisfiability Suggested Format”`_.

   .. _“Satisfiability Suggested Format”: http://www.domagoj-babic.com/uploads/ResearchProjects/Spear/dimacs-cnf.pdf

The DIMACS input file is a plain ASCII text file. It contains lines of several
types described below.
A line is terminated with an end-of-line character.
Fields in each line are separated by at least one blank space.

Comment lines
"""""""""""""
Comment lines give human-readable information about the file and are ignored by
programs.
Comment lines can appear anywhere in the file.
Each comment line begins with a lower-case character `c`.

::

    c This is a comment line

Problem line
""""""""""""
There is one problem line per data file.
The problem line must appear before any clause lines.
It has the following format::

    p cnf VARIABLES CLAUSES

The lower-case character `p` signifies that this is a problem line.
The three character problem designator `cnf` identifies the file as
containing specification information for the CNF-SAT problem.
The `VARIABLES` field contains an integer value specifying :math:`n`, the
number of variables in the instance.
The `CLAUSES` field contains an integer value specifying :math:`m`, the number
of clauses in the instance.

Clauses
"""""""
The clauses appear immediately after the problem line.
The variables are assumed to be numbered from 1 up to :math:`n`.
It is not necessary that every variable appears in the instance.
Each clause is represented by a sequence of numbers separated by either a
space, tab, or new-line character.
The non-negated version of a variable :math:`j` is represented by :math:`j`;
the negated version is represented by :math:`-j`.
Each clause is terminated by the value 0. Unlike many formats that represent
the end of a clause by a new-line character, this format allows clauses to be
on multiple lines.

Example
"""""""
Below here is an example of the data file in DIMACS format corresponding to the
CNF-SAT problem :eq:`1.2`.

::

    c sample.cnf
    c
    c This is an example of the CNF-SAT problem data
    c in DIMACS format.
    c
    p cnf 4 3
    1 2 0
    -4 3
    -2 0
    -1 4 0
    c
    c eof


GLPK API Routines
=================

.. automethod:: Problem.read_cnfsat

.. automethod:: Problem.check_cnfsat

.. automethod:: Problem.write_cnfsat

.. automethod:: Problem.minisat1

.. automethod:: Problem.intfeas1
