# TO USE:

#df['real_feel'] = df[['Air temperature (°C)','Air humidity (%)','Windspeed (km/h)']].apply(lambda x:feels_like(temperature=Temp(x[0], 'c'), humidity=x[1], wind_speed=x[2]).c, axis=1)
#df['wind_chill'] = df[['Air temperature (°C)','Air humidity (%)','Windspeed (km/h)']].apply(lambda x:wind_chill(temperature=Temp(x[0], 'c'), wind_speed=x[2]).c, axis=1)
#df['heat_index'] = df[['Air temperature (°C)','Air humidity (%)','Windspeed (km/h)']].apply(lambda x:heat_index(temperature=Temp(x[0], 'c'), humidity=x[1]).c, axis=1)
#df['dew_point'] = df[['Air temperature (°C)','Air humidity (%)','Windspeed (km/h)']].apply(lambda x:dew_point(temperature=Temp(x[0], 'c'), humidity=x[1]).c, axis=1)

import math
from functools import wraps
import operator

class FloatCompatible(type):
    """Metaclass to make Temp class compatible with float for basic math.

    This will allow to mix Temp class with floats in basic math expressions
    and return Temp instance in result of the same unit.
    """

    math_methods = [
        '__add__', '__sub__', '__mul__', '__truediv__',
        '__pos__', '__neg__',
    ]
    math_rmethods = ['__radd__', '__rsub__', '__rmul__', '__rtruediv__']

    def __new__(cls, name, bases, namespace):

        for method in cls.math_methods:
            namespace[method] = cls.math_method(method)

        for rmethod in cls.math_rmethods:
            method = rmethod.replace('__r', '__')
            namespace[rmethod] = cls.math_method(method, right_operator=True)

        return super(FloatCompatible, cls).__new__(cls, name, bases, namespace)

    @classmethod
    def math_method(cls, name, right_operator=False):
        """Generate method for math operation by name.

        :param name: name of method. i.e. '__add__'
        :param right_operator: is it a "right" operation as '__radd__'
        :type right_operator: bool
        """

        math_func = getattr(operator, name)

        @wraps(math_func)
        def wrapper(*args):
            # [self, other] - binary operators, [self] - unary
            args = list(args)

            self = args[0]
            args[0] = self.value

            if right_operator:
                args = args[::-1]  # args: self, other -> other, self

            result = math_func(*args)
            return type(self)(result, unit=self.unit)

        return wrapper
    
    


def dew_point(temperature, humidity):
    """Calculate Dew Point temperature.

    Two set of constants are used provided by Arden Buck: for positive and
    negative temperature ranges.

    :param temperature: temperature value in Celsius or Temp instance.
    :type temperature: int, float, Temp
    :param humidity: relative humidity in % (1-100)
    :type humidity: int, float
    :returns: Dew Point temperature
    :rtype: Temp
    """

    CONSTANTS = dict(
        positive=dict(b=17.368, c=238.88),
        negative=dict(b=17.966, c=247.15),
    )

    if humidity < 1 or humidity > 100:
        msg = 'Incorrect value for humidity: "{}". Correct range 1-100.'
        raise ValueError(msg.format(humidity))

    T = temperature.c if isinstance(temperature, Temp) else temperature
    RH = humidity

    const = CONSTANTS['positive'] if T > 0 else CONSTANTS['negative']

    pa = RH / 100. * math.exp(const['b'] * T / (const['c'] + T))

    dp = const['c'] * math.log(pa) / (const['b'] - math.log(pa))

    return Temp(dp, C)


def heat_index(temperature, humidity):
    """Calculate Heat Index (feels like temperature) based on NOAA equation.

    HI is useful only when the temperature is minimally 80 F with a relative
    humidity of >= 40%.

    Default unit for resulting Temp value is Fahrenheit and it will be used
    in case of casting to int/float. Use Temp properties to convert result to
    Celsius (Temp.c) or Kelvin (Temp.k).

    :param temperature: temperature value in Fahrenheit or Temp instance.
    :type temperature: int, float, Temp
    :param humidity: relative humidity in % (1-100)
    :type humidity: int, float
    :returns: Heat Index value
    :rtype: Temp
    """

    c1 = -42.379
    c2 = 2.04901523
    c3 = 10.14333127
    c4 = -0.22475541
    c5 = -6.83783e-3
    c6 = -5.481717e-2
    c7 = 1.22874e-3
    c8 = 8.5282e-4
    c9 = -1.99e-6

    T = temperature.f if isinstance(temperature, Temp) else temperature
    RH = humidity

    # try simplified formula first (used for HI < 80)
    HI = 0.5 * (T + 61. + (T - 68.) * 1.2 + RH * 0.094)

    if HI >= 80:
        # use Rothfusz regression
        HI = math.fsum([
            c1,
            c2 * T,
            c3 * RH,
            c4 * T * RH,
            c5 * T**2,
            c6 * RH**2,
            c7 * T**2 * RH,
            c8 * T * RH**2,
            c9 * T**2 * RH**2,
        ])

    return Temp(HI, unit=F)



C = 'c'  # Celsius
F = 'f'  # Fahrenheit
K = 'k'  # Kelvin


# support metaclass both in Python 2 and 3
AbstractTemp = FloatCompatible('AbstractTemp', (object, ), {})


class Temp(AbstractTemp):
    """Temperature value.

    Temp instance can be created in any unit by specifying `unit` attribute.
    Can be converted to any unit by using properties: .c. .f, .k

    Currently supported units:
        C - Celsius
        F - Fahrenheit
        K - Kelvin
    """

    _allowed_units = (C, F, K)
    _conversions = dict(
        # Fahrenheit
        c2f=lambda t: t * 9 / 5. + 32,
        f2c=lambda t: (t - 32) * 5 / 9.,
        # Kelvin
        c2k=lambda t: t + 273.15,
        k2c=lambda t: t - 273.15,
    )

    def __init__(self, temperature, unit='C'):
        """Create new temperature value.

        :param temperature: temperature value in selected units.
        :type temperature: int, float
        :param unit: temperature unit, allowed values: C, F, K.
        :type unit: str
        """

        self.unit = unit.lower()
        self.value = float(temperature)

        if self.unit not in self._allowed_units:
            allowed_units = ', '.join(
                map(lambda u: '"%s"' % u.upper(), self._allowed_units)
            )
            msg = 'Unsupported unit "{}". Currently supported units: {}.'
            raise ValueError(msg.format(unit, allowed_units))

    @classmethod
    def convert(cls, value, from_units, to_units):
        """Convert temperature value between any supported units.

        Conversion is performed using Celsius as a base unit.
        i.e. Fahrenheit -> Kelvin will be converted in two steps: F -> C -> K

        :param value: temperature value
        :type value: int, float
        :param from_units: source units ('C', 'F', 'K')
        :param to_units: target units ('C', 'F', 'K')
        :rtype: float
        """

        from_units = from_units.lower()
        to_units = to_units.lower()

        if from_units == to_units:
            return value

        if from_units != C:
            func_name = '{}2{}'.format(from_units, C)
            f = cls._conversions[func_name]
            value = f(value)

            if to_units == C:
                return value

        func_name = '{}2{}'.format(C, to_units)
        f = cls._conversions[func_name]
        return f(value)

    def _convert_to(self, unit):
        return self.convert(self.value, from_units=self.unit, to_units=unit)

    @property
    def c(self):
        """Temperature in Celsius."""
        return self._convert_to(C)

    @property
    def f(self):
        """Temperature in Fahrenheit."""
        return self._convert_to(F)

    @property
    def k(self):
        """Temperature in Kelvin."""
        return self._convert_to(K)

    def __float__(self):
        return self.value

    def __int__(self):
        return int(self.value)

    def __round__(self, n=0):
        return round(self.value, n)

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return 'Temp({}, unit="{}")'.format(self.value, self.unit.upper())
    
    
def feels_like(temperature, humidity, wind_speed):
    """Calculate Feels Like temperature based on NOAA.

    Logic:
    * Wind Chill: temperature <= 50 F and wind > 3 mph
    * Heat Index: temperature >= 80 F
    * Temperature as is: all other cases

    Default unit for resulting Temp value is Fahrenheit and it will be used
    in case of casting to int/float. Use Temp properties to convert result to
    Celsius (Temp.c) or Kelvin (Temp.k).

    :param temperature: temperature value in Fahrenheit or Temp instance.
    :type temperature: int, float, Temp
    :param humidity: relative humidity in % (1-100)
    :type humidity: int, float
    :param wind_speed: wind speed in mph
    :type wind_speed: int, float
    :returns: Feels Like value
    :rtype: Temp

    """

    T = temperature.f if isinstance(temperature, Temp) else temperature

    if T <= 50 and wind_speed > 3:
        # Wind Chill for low temp cases (and wind)
        FEELS_LIKE = wind_chill(T, wind_speed)
    elif T >= 80:
        # Heat Index for High temp cases
        FEELS_LIKE = heat_index(T, humidity)
    else:
        FEELS_LIKE = T

    return Temp(FEELS_LIKE, unit=F)

def wind_chill(temperature, wind_speed):
    """Calculate Wind Chill (feels like temperature) based on NOAA.

    Default unit for resulting Temp value is Fahrenheit and it will be used
    in case of casting to int/float. Use Temp properties to convert result to
    Celsius (Temp.c) or Kelvin (Temp.k).

    Wind Chill Temperature is only defined for temperatures at or below
    50 F and wind speeds above 3 mph.

    :param temperature: temperature value in Fahrenheit or Temp instance.
    :type temperature: int, float, Temp
    :param wind_speed: wind speed in mph
    :type wind_speed: int, float
    :returns: Wind chill value
    :rtype: Temp
    """

    T = temperature.f if isinstance(temperature, Temp) else temperature
    V = wind_speed

    if T > 50 or V <= 3:
        return Temp(0, unit=C)
    else:

        WINDCHILL = 35.74 + (0.6215 * T) - 35.75 * V**0.16 + 0.4275 * T * V**0.16

        return Temp(WINDCHILL, unit=F)
