/// LFSR-based random number generator matching the HPCC C reference.
///
/// Constants from RandomAccess.h:
///   POLY   = 0x0000000000000007
///   PERIOD = 1317624576693539401

const POLY: i64 = 0x0000000000000007;
const PERIOD: u64 = 1317624576693539401;

/// Single LFSR step: `ran = (ran << 1) ^ (ran < 0 ? POLY : 0)`
#[inline]
pub fn lfsr_step(ran: i64) -> i64 {
    (ran << 1) ^ if ran < 0 { POLY } else { 0 }
}

/// Advance the LFSR by `n` steps using exponentiation by squaring.
///
/// This exactly replicates the C `starts(n)` function from the HPCC reference.
/// Returns the LFSR state after advancing `n` steps from the initial state 1.
pub fn starts(n: u64) -> i64 {
    let mut n = n;
    // Normalize n into [0, PERIOD)
    while n >= PERIOD {
        if n < PERIOD {
            break;
        }
        n -= PERIOD;
    }

    if n == 0 {
        return 0x1;
    }

    // Precompute m2[i] = LFSR state after 2^(i+1) steps from 1
    // m2[i] represents the state after 2^(i+1) steps
    let mut m2: [u64; 64] = [0; 64];
    let mut temp: u64 = 0x1;
    for i in 0..64 {
        m2[i] = temp;
        temp = lfsr_step(temp as i64) as u64;
        temp = lfsr_step(temp as i64) as u64;
    }

    // Find the highest bit set in n (starting from bit 62)
    let mut i: i32 = 62;
    while i >= 0 {
        if (n >> i) & 1 != 0 {
            break;
        }
        i -= 1;
    }

    // Start from step 1 (ran = 2, which is the state after 1 step)
    let mut ran: u64 = 0x2;

    while i > 0 {
        // Compute ran = m2[0]*ran_bit0 ^ m2[1]*ran_bit1 ^ ... ^ m2[63]*ran_bit63
        let mut temp: u64 = 0;
        for j in 0..64 {
            if (ran >> j) & 1 != 0 {
                temp ^= m2[j];
            }
        }
        ran = temp;
        i -= 1;
        if (n >> i) & 1 != 0 {
            ran = lfsr_step(ran as i64) as u64;
        }
    }

    ran as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lfsr_step() {
        // Starting from 1: (1 << 1) ^ 0 = 2
        assert_eq!(lfsr_step(1), 2);
        // From 2: (2 << 1) ^ 0 = 4
        assert_eq!(lfsr_step(2), 4);
    }

    #[test]
    fn test_starts_zero() {
        assert_eq!(starts(0), 0x1);
    }

    #[test]
    fn test_starts_one() {
        // After 1 step from initial 1: lfsr_step(1) = 2
        assert_eq!(starts(1), 2i64);
    }

    #[test]
    fn test_starts_period() {
        // After PERIOD steps we should be back to 1
        assert_eq!(starts(PERIOD), 0x1);
    }

    #[test]
    fn test_starts_consistency() {
        // starts(a+b) should equal stepping starts(a) by b steps
        let a: u64 = 12345;
        let b: u64 = 67890;
        let combined = starts(a + b);
        let step_a = starts(a);
        let mut stepped = step_a;
        for _ in 0..b {
            stepped = lfsr_step(stepped);
        }
        assert_eq!(combined, stepped);
    }
}
