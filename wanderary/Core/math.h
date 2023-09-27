#ifndef WDR_MATH_H_
#define WDR_MATH_H_

namespace wdr::math
{
  // Refer to  https://github.com/simonfuhrmann/mve/blob/master/libs/math/accum.h
  template <typename T>
  class Accumulator
  {
  public:
    T v;
    float w;

  public:
    /** Leaves internal value uninitialized. */
    Accumulator(void);

    /** Initializes the internal value (usually to zero). */
    Accumulator(T const &init);

    /** Adds the weighted given value to the internal value. */
    void add(T const &value, float weight);

    /** Subtracts the weighted given value from the internal value. */
    void sub(T const &value, float weight);

    /**
     * Returns a normalized version of the internal value,
     * i.e. dividing the internal value by the given weight.
     * The internal value is not changed by this operation.
     */
    T normalized(float weight) const;

    /**
     * Returns a normalized version of the internal value,
     * i.e. dividing the internal value by the internal weight,
     * which is the cumulative weight from the 'add' calls.
     */
    T normalized(void) const;
  };
}

#endif // WDR_MATH_H_