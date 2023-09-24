/**
 * Common data types that are used in the source code. 
 * Author: Guoxi Liu <guoxil@clemson.edu>
 *         Guillaume Favelier <guillaume.favelier@lip6.fr>
 * Date: 2023/09/20
 * 
*/

#pragma once

/**
 * @brief Mark function/method parameters that are not used in the
 * function body at all.
 *
 * This is the TTK version of vtkNotUsed. It silences compiler
 * warnings about unused parameters when those parameters are not used
 * in the current function/method body but still necessary in the
 * function/method signature (for instance when implementing a
 * particular API with inheritance).
 *
 * Basically, it removes the name of the parameter at its definition
 * site (similar to commenting the parameter name).
 */
#define ttkNotUsed(x)

namespace ttk {
  using LongSimplexId = long long int;
  using SimplexId = long long int;

}
