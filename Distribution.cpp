#include "Distribution.h"

double Distribution::getProbability( int n, int v ) {
	return 1.0 / ( 5.0 + n * v  );
}
