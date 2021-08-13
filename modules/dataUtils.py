import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import namedtuple
from pandas.api.types import CategoricalDtype
import matplotlib, datetime, platform, math, scipy, sklearn, xlrd
import os
from scipy import stats
# xlrd is needed for opening excel files with pandas, not installed automatically
from sklearn.linear_model import LinearRegression
from abc import ABC, abstractmethod
from IPython.display import display, Latex, Markdown

md = lambda x: display(Markdown(x))

# dataUtils Vars

user = ""
password = ""
port = ""

mode = "remote"

yesTuple = ("YES", "Yes", "yes", "Y", "Yes ", "yes ", "YES ", "y")
noTuple = ("NO", "No", "no", "N", "n")
nanTuple = (np.nan, float("Nan"))

dataDropKey = {
    "EMPTY": "Empty Column",
    "PII": "Personally Identifying Information",
    "DD": "Redundant/Duplicate Data",
    "UNF": "Unformatted Data (Reformatted Elsewhere)",
}

# write out explicitly for auto completion
DD = dataDropKey["DD"]
EMPTY = dataDropKey["EMPTY"]
PII = dataDropKey["PII"]
UNF = dataDropKey["UNF"]

ignoreDrop = []

logFilename = "tarch.log"


def initLogging():
    """
    Initializes logging, adds an explicit logging level for 'DATADROP'

    :return: logger object
    """

    def dataDropLogger(self, message, *args, **kws):
        # Pass through function to enable an explicit "DATADROP" logging level
        if self.isEnabledFor(DEBUG_LEVELV_NUM):
            # Yes, logger takes its '*args' as 'args'.
            self._log(DEBUG_LEVELV_NUM, message, args, **kws)

    DEBUG_LEVELV_NUM = 35  # between WARNING and ERROR
    logging.addLevelName(DEBUG_LEVELV_NUM, "DATADROP")

    logging.Logger.dataDropLogger = dataDropLogger

    logging.basicConfig(level=logging.DEBUG,
                        filename=logFilename,
                        format='%(asctime)s.%(msecs)03d %(name)-25s %(levelname)-7s %(message)s',
                        datefmt='%y-%m-%d %H:%M:%S',
                        filemode="w")

    loggerFileHandler = logging.FileHandler(logFilename)
    loggerFileHandler.setLevel(logging.DEBUG)

    loggerConsoleHandler = logging.StreamHandler()
    loggerConsoleHandler.setLevel(logging.INFO)
    loggerConsoleHandler.setFormatter(logging.Formatter("%(levelname)-8s - %(name)-12s - %(message)s"))

    logging.getLogger('').addHandler(loggerFileHandler)
    logging.getLogger('').addHandler(loggerConsoleHandler)

    return logging.getLogger("tarch-main")


l = initLogging()


def getLogger(module, forceString=False):
    """
    Returns logger based on module name

    :param module: ostensibly any object, written with utils.DataModule in mind
    :param forceString: use if module.name() isn't implemented / doesn't return a string
    :type forceString: bool
    :return:
    """
    name = str(module) if forceString else module.name()
    return logging.getLogger(name)


def verifyVersioning():
    """
    Verifies versions of important packages and python installation, raises exception and quits if versions aren't
    correct

    :return:
    """
    versions = {pd: "1.0.3", sns: "0.10.0", np: "1.18.3", matplotlib: "3.2.1", scipy: "1.4.1",
                sklearn: "0.22.2.post1", xlrd: "1.2.0"}
    pythonVersion = "3.8.2"
    for package in versions:
        l.debug("{package} version is {version}.".format(package=package.__name__, version=package.__version__))
        try:
            assert package.__version__ == versions[package]
        except AssertionError:
            l.critical("{package} not required version {version}. Version {actVersion} found instead."
                       .format(package=package.__name__, version=versions[package], actVersion=package.__version__))
            exit(1)
    l.debug("Python version: {0}".format(platform.python_version()))
    try:
        assert platform.python_version() == pythonVersion
    except AssertionError:
        l.critical("Python version not required version {0}. Version {actVersion} found instead.".format(pythonVersion,
                                                                                                         actVersion=platform.python_version()))
        exit(1)

class ImpliedBool(object):
    """
    Used for variables that we want to treat as bool in analyses *but* want to flag for some reason

    :param value: bool value to "act" as
    :type value: bool
    :param string: message to raise when printing/displaying. Formatted with value so "eg: {}" becomes "eg: value"
    :type string: str
    """
    def __init__(self, value, string):
        self.value = value
        self.string = string

    def __bool__(self):
        return self.value

    def __nonzero__(self):
        return self.__bool__()

    def __repr__(self):
        return self.string.format(self.value)


ImpliedFalse = ImpliedBool(False, "Implied {}")
ImpliedTrue = ImpliedBool(True, "Implied {}")


class CustomNA(object):
    """
    Used for variables that we want to treat as NA in analyses *but* want to flag for some reason

    :param reason: message to raise when printing/displaying.
    :type reason: str
    """
    def __init__(self, reason):
        self.reason = reason

    def __repr__(self):
        return self.reason


    @staticmethod
    def __array__():
        # for pd.isnull and np.isnan
        return np.array(np.nan)


NotCollected = CustomNA("Not Collected")
NotReported = CustomNA("Not Reported")


NC = NotCollected  # aka championship was not a variable that existed before 2014
iTrue = ImpliedTrue  # aka everyone went to championships after 2017ish, so it's implied that a team went
iFalse = ImpliedFalse
NR = NotReported  # aka one missing data point in an otherwise complete dataset

unf = "_unformatted"


class DataModule(ABC):
    """
    Abstract class that sets the layout for how all data modules must be written.
    """
    def __init__(self):
        self.depends = ()
        self.dataStageFlag = None
        self.run = True
        self.checkPoint = None
        self.l = None

    @property
    @abstractmethod
    def name():
        return

    @property
    @abstractmethod
    def source():
        return

    @property
    @abstractmethod
    def productionReady():
        return

    @abstractmethod
    def importData(self, MasterFrame):
        return

    @abstractmethod
    def prepareData(self):
        return

    @abstractmethod
    def annotateData(self):
        return

    @abstractmethod
    def mergeData(self, MasterFrame):
        return

    @property
    @abstractmethod
    def describeData():
        return

    def setCheckpoint(self):
        self.l.debug("Setting a checkpoint...")
        self.checkPoint = self.data.columns
        self.l.debug("Setting checkpoint set to {}".format(self.checkPoint))

    def logChanges(self):
        setCheckPoint = set(self.checkPoint)
        setCurrent = set(self.data.columns)
        self.l.debug(
            "Changes in columns (by name) since last checkpoint: \n+{0}\n-{1}".format(setCurrent - setCheckPoint,
                                                                                      setCheckPoint - setCurrent))


def splitByString(string, splitString, excludeString, skipLast=False):
    """
    Splits string by splitString unless surronded by exludeString

    :param string: input string to operate on
    :type string: str
    :param splitString: string to split by
    :type splitString: str
    :param excludeString: string to escape splitting
    :type excludeString: str
    :param skipLast: Skip last string
    :type skipLast: bool

    :returns list: string split into substring list elements

    **Examples**

    >>> splitByString("a;b", ";", "%")
    ["a", "b"]
    >>> splitByString("%a;b%", ";", "%")
    ["%a;b%"]
    >>> splitByString("%a;b", ";", "%")
    ["%a;b"]
    >>> splitByString("a;b%", ";", "%")
    ["a", "b%"]
    """
    if splitString == excludeString:
        return Exception("Splitting strings are the same")

    # initailze some state machine working variables
    splitArray = []
    seenAnExcludeString = False
    word = ""

    # state machine iteration over each letter in string
    for letter in string:
        if letter == splitString:
            if not seenAnExcludeString:
                splitArray.append(word)
                word = ""
                continue
        elif letter == excludeString:
            seenAnExcludeString = not seenAnExcludeString
        word = word + letter
    if not skipLast:
        splitArray.append(word)
    return splitArray


def removeQuotesIfThere(quotedString, useSingleQuote=False):
    """ Removes quotes from string.
    Fails silently, only removes quotes if there - if not, just returns quotedString

    :param quotedString: input string to remove quotes from
    :type quotedString: str
    :param useSingleQuote: if True remove single (') quote instead of default False (")
    :type useSingleQuote: bool

    :return string: quotedString with at max two chars removed from the beginning and end.

    **Examples:**

    >>> removeQuotesIfThere('"test"')
    'test'
    >>> removeQuotesIfThere('"test')
    '"test'
    >>> removeQuotesIfThere("'test'")
    "'test'"
    >>> removeQuotesIfThere("'test'", useSingleQuote=True)
    "test"
    """
    if type(quotedString) == str:
        if useSingleQuote:
            if quotedString.startswith("'") and quotedString.endswith("'"):
                return quotedString[1:-1]
        else:
            if quotedString.startswith('"') and quotedString.endswith('"'):
                return quotedString[1:-1]
        return quotedString


def castColumn(df, column, function, logger, name=None, inPlace=True, replaceColumn=False):
    """
    :param df: pandas dataframe
    :type df: pandas.DataFrame
    :param column: column name in df to cast
    :type column: str
    :param function: casting function (anything that can be used via pandas .applymap)
    :type function: function
    :param logger: logging object
    :param name: new name for casted column
    :type name: str
    :param inPlace: default True, modify df in place - if false return new DataFrame
    :type inPlace: bool

    .. note:: depending on value of *inPlace*, return type is different

    if inPlace is True

    :returns: (str) name of new column

    if inPlace is False

    :returns pandas.DataFrame, str: New modified column where each element of df column is mapped to function(element), name of new column

    **Examples**
    
    >>> df = pd.DataFrame({"AAA": [4,5,6,7],
    >>>                    "BBB": [10,20,30,40],
    >>>                    "CCC": [100,50,-30,-50]})
           AAA  BBB  CCC
        0    4   10  100
        1    5   20   50
        2    6   30  -30
        3    7   40  -50
    >>> def f(x):
    >>>     return x*2
    >>>
    >>> castColumn(df.copy(), "AAA", f, logger)
    "f(AAA)"
    >>> castColumn(df.copy(), "AAA", f, logger, name="Double AAA")
    "Double AAA"
    >>> castColumn(df, "AAA", f, logger, inPlace=False)
    (   AAA  BBB  CCC  f(AAA)
     0    4   10  100       8
     1    5   20   50      10
     2    6   30  -30      12
     3    7   40  -50      14,
     'f(AAA)')

    """

    # set name
    if (not replaceColumn) and (name == None and function.__name__ == "<lambda>"):
        raise Exception("Don't use lambdas - we want *real* functions here for readability/auditing!")
    if name == None:
        name = function.__name__ + "(" + column + ")"

    logger.debug("Casting column {c} via function {f} to column with name {n}".format(c=column, f=function, n=name))
    # create copy if we don't want to modify the original df
    dff = df if inPlace else df.copy()
    # actually cast column
    dff[name] = dff[[column]].applymap(function)
    if replaceColumn:
        if column + unf in dff.columns:
            dff.drop(columns=column + unf)
        dff.rename(columns={column: column + unf}, inplace=True)
        dff.rename(columns={name: column}, inplace=True)
        dff = dff.T.drop_duplicates().T
        name = column
    return name if inPlace else (dff, name)



def applyMapRecursively(series, function):
    """
    Apply function to map a iterable to a new list interatively, helper function for castComplicatedColumn (below)

    :param series: input iterable to iterate over and translate to new list
    :type series: list

    :param function: function to apply
    :type function:

    **Examples**

    >>>    testList = [0,1,2,3]
    >>>
    >>>    applyMapRecursively(testList, str)
    ["0", "1", "2", "3"]
    >>>
    >>>    def f(x):
    >>>        isOdd = lambda x: bool(x % 2)
    >>>        if isOdd(x):
    >>>            return lambda x: x*3
    >>>        else:
    >>>            return "Even!"
    >>>
    >>>    applyMapRecursively(testList, f)
    ["Even!", 3, "Even!", 9]
    """
    newSeries = []
    for item in series:
        f = function
        while callable(f): f = f(item)
        newSeries.append(f)
    return newSeries


# new named tuple for readability in passed variables
excludeBy = namedtuple("excludeBy", "colName values storeValue")


def castComplicatedColumn(df, originalCol, newCol, castDict, exclude=None, inPlace=True, tagOldColumn=False):
    '''
    df (DataFrame): dataframe to operate on
    originalCol (string): column to cast from df
    newCol (string): new name for cast column
    castDict (dict): mapping for casting
    exclude (list): list of excludeBy namedtuples to exclude via column and values of said column and replace with storeValue 
    inPlace: modify in place or return new df
    
    returns:
        inPlace: True
            DataFrame: new modified df with cast column newCol
    
    example:
                df =    AAA    BBB    CCC
                   0    True   2      string1
                   1    True   3      string2
                   2    False  4      string3
    
            castComplicatedColumn(df, "AAA", "cast(AAA)", {True: "Yes", False: "No"})
                df =    AAA    BBB    CCC      cast(AAA)
                   0    True   2      string1  Yes
                   1    True   3      string2  Yes 
                   2    False  4      string3  No
            
            castComplicatedColumn(df, "AAA", "cast(BBB)", {2: "Valid", 3:"Valid", 4:"Invalid"}, inPlace=False) => dff
               dff =    AAA    BBB   cast(BBB)   CCC
                   0    True   2     Valid       string1
                   1    True   3     Valid       string2
                   2    False  4     Invalid     string3
            
            
            castComplicatedColumn(df, "CCC", "cast(CCC)", {"string1": "A", "string2": "B", "string3": "C"}, 
                                  exclude=[excludeBy(colName="AAA", values=[False], storeValue=NC)])
                df =    AAA    BBB    CCC      cast(CCC)
                   0    True   2      string1  A
                   1    True   3      string2  B
                   2    False  4      string3  NC
    '''
    # create copy if we don't want to modify the original df
    dff = df if inPlace else df.copy()

    # initalize a new categorical column
    dff[newCol] = np.nan
    categories = list(set(castDict.values()))
    dff[newCol] = dff[newCol].astype(CategoricalDtype(categories=categories))

    # populate new column
    dff[newCol] = applyMapRecursively(dff[originalCol], castDict.get)
    # display(dff[originalCol])

    # Recast to category since we've got three vars: True, False and None 
    # (bool can't hold this in pandas, see https://pandas.pydata.org/pandas-docs/stable/user_guide/gotchas.html#na-type-promotions for more info on casting)
    if exclude:
        for excludeEntry in exclude:
            # add to categories if need be
            if not excludeEntry.storeValue in categories:
                categories.append(excludeEntry.storeValue)
                dff[newCol] = dff[newCol].astype(CategoricalDtype(categories=categories))
            # set excluded entries to passed value
            dff.loc[(dff[excludeEntry.colName].isin(excludeEntry.values)), newCol] = excludeEntry.storeValue
    if tagOldColumn:
        if originalCol + unf in dff.columns:
            dff.drop(columns=originalCol + unf)
        dff.rename(columns={originalCol: originalCol + unf}, inplace=True)
        dff = dff.T.drop_duplicates().T  # remove duplicate columns
    if not inPlace:
        return dff


def messyBoolCaster(string):
    """Takes string and casts to bool
    :param string: input string to cast
    :type string: str

    :return: if string "1", "True" or "true"
    :rtype: bool

    **Examples**

    >>> messyBoolCaster("True")
    True
    >>> messyBoolCaster("1")
    True
    >>> messyBoolCaster("true")
    True
    >>> messyBoolCaster("False")
    False
    >>> messyBoolCaster("blah blah blah")
    False
    """
    return (string in ["1", "True", "true", 1, True])


def getBoolCaster(convBlankTo):
    """ Returns a function that can cast bools and blank strings to bools

    :param convBlankTo: what to convert blanks to in returned caster (ideally bool)

    :returns: takes in string and converts s to boolFromString(s) unless blank, then returns convBlankTo
    :rtype: function

    **Examples:**

    >>> a = getBoolCaster(True)
    >>> a("True")
    True
    >>> a("")
    True

    >>> b = getBoolCaster(False)
    >>> b("True")
    True
    >>> b("")
    False

    >>> c = getBoolCaster("anything can go here")
    >>> c("True")
    True
    >>> c("")
    "anything can go here"

    """

    def f(string):
        if string == "":
            return convBlankTo
        else:
            return messyBoolCaster(string)

    return f


def igemDatabaseJSONToDataFrame(filename, messy=True):
    """ Takes in exported JSON and returns pandas.DataFrame

    :param filename: input .json file
    :type filename: str
    :param messy: some exports are "messy" and have poor delimiter formatting - setting this to True enables import of these files as well
    :type messy: bool

    :returns: dataframe containing JSON data
    :rtype: pandas.DataFrame

    **Examples**

    >>> a.json = 'AAA;BBB;CCC;\\n1;2;"string";' # file contents
    >>> igemDatabaseJSONToDataFrame("a.json")
       AAA  BBB       CCC
    0    1    2    string

    """
    with open(filename, encoding="utf8") as f:
        # read json DB dump
        db = f.read()
    if messy:
        # reformat poorly formatted json to be interperable by pandas
        db = splitByString(db, "\n", excludeString='"', skipLast=True)  # split by linebreaks unless they're quoted out
        db = [splitByString(x, ";", excludeString='"') for x in db]  # items are split by semicolons
        db = [[removeQuotesIfThere(entry) for entry in line] for line in db]
        # create pandas dataframe from list-ified json
        rows = {i: x for i, x in enumerate(db[1:])}
        df = pd.DataFrame.from_dict(rows, orient='index', columns=db[0])
    else:
        df = pd.read_json(db)
    return df


def tagDf(df, tag, logger, exempt=[], inplace=False):
    """ Returns df with 'tagged' column names

    :param df: dataframe to tag
    :type df: pandas.DataFrame
    :param tag: tag to append to all column names
    :type tag: str
    :param exempt: columns to skip when appending tags
    :type exempt: list
    :param inplace: default True, modify df in place - if false return new DataFrame
    :type inplace: bool

    if inPlace is False

    :returns: df with new column names

    **Examples**

    >>> df = pd.DataFrame({"AAA": [4,5,6,7],
    >>>                    "BBB": [10,20,30,40],
    >>>                    "CCC": [100,50,-30,-50]})
    >>> df
           AAA  BBB  CCC
        0    4   10  100
        1    5   20   50
        2    6   30  -30
        3    7   40  -50
    >>> tagDf(df, "_cat")
           AAA_cat  BBB_cat  CCC_cat
        0        4       10      100
        1        5       20       50
        2        6       30      -30
        3        7       40      -50
    >>> tagDf(df, "_new", inplace=True) # doesn't return anything
    >>> df
           AAA_new  BBB_new  CCC_new
        0        4       10      100
        1        5       20       50
        2        6       30      -30
        3        7       40      -50

    """
    dff = df if inplace else df.copy()
    if inplace:
        logger.info(f"Tagging data with tag '{tag}'")
    for col in dff:
        if not col in exempt:
            dff.rename(columns={col: col + tag}, inplace=True)
    if not inplace:
        return dff


def expDict(dic):
    """ Flattens dictionary objects

    :param dic: input dictonary (eg. {[1,"Yes"]: True, [0, "No"]: False})
    :type dic: dict

    :return: expanded dict, see examples below for clarity
    :rtype: dict

    **Examples:**

    >>> expDict({(1, "Yes"): True})
    {1: True, "Yes": True}
    >>> expDict({1: "a"})
    {1: a}
    >>> expDict({1: ("a", "b")})
    {1: ("a", "b")}
    >>> expDict({(1,2):True, 1:False})
    RuntimeError("Overlapping Keys")

    """
    newDict = {}
    for key in dic:
        if type(key) == tuple:
            item = dic[key]
            for entry in key:
                newDict[entry] = item
        else:
            newDict[key] = dic[key]
    return newDict

def stripHandleNan(input):
    """
    Just like str.strip but returns np.nan if input is in ["nan", "NaN", "Nan"]

    :param input: input string or Nan (np.nan or NaN handled)
    :type input: str, float

    **Examples**

    >>> upperHandleNan("hi")
    "HI"
    >>> upperHandleNan(np.nan)
    np.nan
    >>> upperHandleNan(float("Nan"))
    np.nan

    """
    if type(input) == str:
        return input.strip()
    elif input == np.nan or input in ["nan", "NaN", "Nan"] or math.isnan(input):
        return np.nan
    raise Exception("Could not strip {0}, type {1}".format(input, type(input)))

def upperHandleNan(input):
    """
    Just like str.upper but returns np.nan if input is in ["nan", "NaN", "Nan"]

    :param input: input string or Nan (np.nan or NaN handled)
    :type input: str, float

    **Examples**

    >>> upperHandleNan("hi")
    "HI"
    >>> upperHandleNan(np.nan)
    np.nan
    >>> upperHandleNan(float("Nan"))
    np.nan

    """
    if type(input) == str:
        return input.upper()
    elif input == np.nan or input in ["nan", "NaN", "Nan"] or math.isnan(input):
        return np.nan
    raise Exception("Could not cast {0}, type {1}".format(input, type(input)))


def nanableOr(x, y):
    """ Or function that can take Nan values

    :param x: input one
    :param y: input two
    :type x: bool, nan
    :type y: bool, nan

    :returns: bool if both nan, x if y is nan, y if x is nan else x or y (see table below)

    +--------------------+----------------------+
    |                    |      Value of X      |
    |                    +------+-------+-------+
    |                    | True | False | Nan   |
    +------------+-------+------+-------+-------+
    | Value of Y | True  | True | True  | True  |
    |            +-------+------+-------+-------+
    |            | False | True | False | False |
    |            +-------+------+-------+-------+
    |            | Nan   | True | False | Nan   |
    +------------+-------+------+-------+-------+

    """
    xn = x in nanTuple
    yn = y in nanTuple

    if xn and yn:
        return np.nan
    elif xn:
        return y
    elif yn:
        return x
    else:
        return x or y


def nanableNot(x):
    """Not function that can take Nan values

    :param x: input
    :type x: bool, NaN

    :return: np.nan if x is (some form of) nan else not x

    **Examples**

    >>> nanableNot(True)
    False
    >>> nanableNot(False)
    True
    >>> nanableNot(np.nan)
    np.nan
    >>> nanableNot(float("Nan"))
    np.nan

    """
    return np.nan if x in nanTuple else not x


def combineBoolCols(df, cols, function=nanableOr):
    """
    df (DataFrame): input dataframe
    cols (list): list of column names of cols in df to merge
    function (function, optional): function to use to merge columns, must take two arguments

    returns: (list) new, merged column

    examples:

        df =    AAA   BBB   CCC
           0    False False True
           1    False True  False
           2    True  False False
           3    False True  True

    combineBoolCols(df, ["AAA", "BBB"]) => [False, True, True, True]
    combineBoolCols(df, ["CCC", "BBB"]) => [True, True, False, True]

    combineBoolCols(df, ["AAA", "BBB", "CCC"]) => [True, True, True, True]

    combineBoolCols(df, ["AAA", "BBB"], function=lambda x,y: True) => [True, True, True, True]
    combineBoolCols(df, ["AAA", "BBB"], function=lambda x,y: "Pineapple") => ["Pineapple", "Pineapple", "Pineapple", "Pineapple"]


    """
    mergeable = [df[x] for x in cols]
    while (len(mergeable) > 1):
        series1 = mergeable.pop()
        series2 = mergeable.pop()
        newSeries = series1.combine(series2, function)
        mergeable.append(newSeries)

    return newSeries


# In[104]:


def settify(df, cols):
    """Returns set of possible values ordered by order of cols

    :param df: input dataframe
    :type df: pandas.DataFrame
    :param cols: col(s) to settify
    :type cols: list, str

    :returns: of possible values ordered by order of cols (see examples)
    :rtype: set

    **Examples**

    >>> df = pd.DataFrame({"AAA": ["Cat","Cat","Dog","Dog"],
    >>>                    "BBB": ["Black","Black","White","Grey"],
    >>>                    "CCC": [0,5,10,10]})
    >>> df
             AAA   BBB   CCC
        0    Cat   Black 0
        1    Cat   Black 5
        2    Dog   White 10
        3    Dog   Grey  10
    >>> settify(df, ["AAA"])
    {"Cat", "Dog"}
    >>> settify(df, ["BBB"])
    {"Black", "White", "Grey"}
    >>> settify(df, ["AAA", "BBB"])
    {("Cat", "Black"), ("Dog", "White"), ("Dog", "Grey")}
    >>> settify(df, ["AAA", "CCC"])
    {("Cat", 0), ("Cat", 5), ("Dog", 10)}
    >>> settify(df, ["AAA", "CCC", "BBB"])
    {("Cat", 0, "Black"), ("Cat", 5, "Black"), ("Dog", 10, "White"), ("Dog", 10, "Grey")}

    """
    if type(cols) == list:
        returnSet = set()
        for index, row in df[cols].iterrows():
            preTuple = []
            for item in cols:
                preTuple.append(row[item])
            returnSet.add(tuple(preTuple))
        return returnSet
    elif type(cols) == str:
        return set(list(df[cols]))


def castColumnsToBool(df, suspectBoolCols, logger):
    """ Safely casts columns to bool, logs actions

    :param df: input dataframe
    :type df: pandas.DataFrame
    :param suspectBoolCols: columns suspected to contain bool-like values
    :type suspectBoolCols: list(str)
    :param logger: logging logger object
    """
    logger.info("Attempting to cast columns {} to bool".format(suspectBoolCols))

    looksLikeBool = []

    # check if things only have 0,1 as values
    for col in suspectBoolCols:
        values = settify(df, col)
        colLooksLikeBool = True if (
                values in ({1}, {0}, {"0"}, {"1"}, {1, 0}, {"1", "0"})) else False  # aka only has 0, 1 as values
        if not colLooksLikeBool:
            logger.info(
                "Guessed that {col} should be type bool (aka has 0, 1 as values). However, it has {values} instead.".format(
                    col=col, values=values))
        looksLikeBool.append(colLooksLikeBool)
    # cast things that look right from above
    for i, looksLikeBoolEntry in enumerate(looksLikeBool):
        if looksLikeBoolEntry:
            castColumn(df, suspectBoolCols[i], messyBoolCaster, logger, name="bool(" + suspectBoolCols[i] + ")")


def castColumnsToUpper(df, cols, logger, handleNan=True):
    """ Safely casts columns to DateTime, logs actions

    :param df: input dataframe
    :type df: pandas.DataFrame
    :param cols: columns with str entries (or float("Nan")) to cast
    :type cols: list(str)
    :param logger: logging logger object
    :param handleNan: if True, use upperHandleNan to cast
    :type handleNan: bool

    """
    logger.info("Attempting to cast columns {0} to upper{1}".format(cols,
                                                                    " and handling nan with upperHandleNan" if handleNan else ""))
    caster = upperHandleNan if handleNan else str.upper
    for col in cols:
        castColumn(df, col, caster, name="upper(" + col + ")", logger=logger)


def castColumnsToDateTime(df, suspectDateCols, logger):
    """ Safely casts columns to DateTime, logs actions

    :param df: input dataframe
    :type df: pandas.DataFrame
    :param suspectDateCols: columns suspected to contain DateTime-like values
    :type suspectDateCols: list(str)
    :param logger: logging logger object
    """
    logger.info("Attempting to cast columns {} to DateTime".format(suspectDateCols))
    for col in suspectDateCols:
        name = "date(" + col + ")"
        df[name] = pd.to_datetime(df[col], errors="raise", exact=True)


def castColumnsToTypeConstructor(caster, castString="", usePandasCasting=False):
    """ Function to generate safe caster objects
    
    Used to generate `castColumnsToCategory` and `castColumnsToInt` functions.
    
    :param caster: function for casting column entries (eg, str. upper or int)
    :type caster: function
    :param castString:
    :type castString: str
    :param usePandasCasting:
    :type usePandasCasting: bool

    :return: column caster
    :rtype: function

    """
    if castString:
        castString = str(caster)
        if usePandasCasting:
            castString = "pandas " + castString

    def function(df, suspectCols, logger):
        logger.info("Attempting to cast columns {0} to {1}".format(suspectCols, castString))
        for col in suspectCols:
            if usePandasCasting:
                df[col] = df[col].astype(caster)
            else:
                castColumn(df, col, caster, logger)

    return function


castColumnsToInt = castColumnsToTypeConstructor(int, "int")
castColumnsToCategory = castColumnsToTypeConstructor("category", "uppercase", usePandasCasting=True)


def safeDropDuplicates(df, cols, logger, reason, reasontext=" ", *args, **kwargs):
    """
    Drops duplicate columns with logging unless --no-drop flag has been explicitly set to ignore.

    :param df: input dataframe
    :type df: pandas.DataFrame
    :param cols: columns to drop
    :type cols: list (of strings)
    :param logger: logger object
    :param reason: reason for dropping (many standard options defined in dataUtils)
    :type reason: str
    :param reasontext: extended reason for dropping, a place for notes to future you
    :type reasontext: str
    :param args: args to pass to pd.DataFrame.drop_duplicates
    :param kwargs: kwargs to pass to pd.DataFrame.drop_duplicates
    :return: None
    """
    if reason in ignoreDrop:
        logger.debug(f"Ignoring request to drop data because {reason} due to flag `--no-drop {reason}`")
        return
    else:
        logger.dataDropLogger(f"Dropping duplicate columns {cols} due to {reason}. {reasontext}"
                              f"Use flag --no-drop {reason} to ignore.")
    df.drop_duplicates(cols, *args, **kwargs)


def safeDropRowByColValues(df, col, values, logger, reason, reasontext=" "):
    """
    Drops rows with specific values in col (with logging) unless --no-drop flag has been explicitly set to ignore.

    :param df: input dataframe
    :type df: pandas.DataFrame
    :param col: column to search
    :type col: str
    :param values: values to search for
    :type values: list
    :param logger: logger object
    :param reason: reason for dropping (many standard options defined in dataUtils)
    :type reason: str
    :param reasontext: extended reason for dropping, a place for notes to future you
    :type reasontext: str
    """

    if reason in ignoreDrop:
        logger.debug(f"Ignoring request to drop data because is {reason} due to flag --no-drop {reason}. {reasontext}")
        return
    else:
        logger.dataDropLogger(f"Dropping data if column {col} has values {values} due to {reason}. {reasontext} "
                              f"Use flag --no-drop {reason} to ignore.")
    for value in values:
        df = df[df[col] != value]


def safeDropColumn(df, col, logger, reason, reasontext=" "):
    """
    Drops rows with specific values in col (with logging) unless --no-drop flag has been explicitly set to ignore.

    :param df: input dataframe
    :type df: pandas.DataFrame
    :param col: column to drop
    :type col: str
    :param logger: logger object
    :param reason: reason for dropping (many standard options defined in dataUtils)
    :type reason: str
    :param reasontext: extended reason for dropping, a place for notes to future you
    :type reasontext: str
    """

    if reason in ignoreDrop:
        logger.debug(f"Ignoring request to drop data because is {reason} due to flag --no-drop {reason} {'' if reasontext==' ' else ', '+reasontext}")
        return
    else:
        logger.dataDropLogger(f"Dropping column {col} due to {reason}. "
                              f"Use flag --no-drop {reason} to ignore and keep the data.")
    df.drop([col], axis=1, inplace=True)



#TODO: FIGURE OUT WHAT TO DO WITH THIS SECTION ;; IS THIS TARCH? OR PALMER GROUP SPECIFIC??

def getFile(filename):
    if mode in ["local", "remote"]:
        return "./data/" + filename
    else:
        # TODO: re-write once tardigrade is up and running, this is a stand in function rn
        from pathlib import Path as path
        import os
        return os.path.join(getGroupPath("box-data/"), filename)


def getGroupPath(filename):
    import os
    return os.path.join(os.environ["GROUP_HOME"], filename)


def sanitizePath(path):
    path = path.replace(" ", "-").replace("*", "ALL")
    return "".join(c for c in path if c.isalnum() or c in "-_.")


def getDBCredentials():
    import os
    return os.environ["IGUSER"], os.environ["IGPASS"], os.environ["IGPORT"]



def queryDatabase(query, database, host="127.0.0.1"):
    """
    Queries database and saves data to a local pickle OR if mode is local, read query data from pickle

    :param query: SQL Query
    :type query: str
    :param database: database to query
    :type database: str
    :param host: database host, default 127.0.0.1
    :type host: str
    :return:
    """
    pickle = getFile(sanitizePath(f"{query}__{database}.query.pickle"))
    if not mode == "local":
        import mysql.connector
        user, password, port = getDBCredentials()
        conn = mysql.connector.connect(
            user=user,
            password=password,
            host=host,
            port=port,
            database=database)
        db = pd.read_sql_query(query, conn)
        db.to_pickle(pickle)
        conn.close()
        return db
    else:
        return pd.read_pickle(pickle)


def safeDropNARows(df, logger):
    """ safely drop all NA rows

    :param df:
    :type df: pandas.DataFrame
    :param logger:
    :type logger: logger object
    """
    before = len(df)
    df = df.dropna(how="all")
    after = len(df)
    logger.info("Successfully dropped {} all NA rows".format(before-after))
    return df


def safeDropEmptyCols(df, logger):
    """ safely drop empty cols (log to info level columns that are dropped)

    An empty column is one that only contains **one** of the following: "", " " or None

    :param df:
    :type df: pandas.DataFrame
    :param logger:
    :type logger: logger object
    """
    logger.info("Looking for empty columns to drop...")

    dropCols = []
    for col in df.columns:
        colSet = list(df[col].unique())
        if colSet in [[], [""], [" "]]:  # we're defining all of these as empty
            dropCols.append(col)

    df.drop(columns=dropCols)
    logger.info("Successfully dropped the following empty columns: {}".format(dropCols))
    return df


def fillBlanksWithNa(df, logger):
    """ replaces '' with np.nan

    :param df:
    :type df: pandas.DataFrame
    :param logger:
    :type logger: logger object
    """
    logger.info("Casting '' to nan")
    df.replace("", np.nan, inplace=True)


def getDescriptions(module):
    """ pulls column descriptions from xlsx/csv files and returns a dict

    :type module: DataModule()
    :return: column -> dict{"description": text description, "follow-up": needs follow-up, "questions": pending questions, "assumptions": assumptions}
    :rtype: dict
    """
    path = f"./../data/descriptors/{module.name()}."
    try:
        df = pd.read_excel(path + "xlsx")
    except FileNotFoundError:
        try:
            df = pd.read_csv(path + "csv")
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find csv or xlsx with file name {path}(csv/xlsx).")
    return {row["Column"]: {"description": row.fillna("")["Description"],
                            "follow-up": pd.isna(row["Needs follow up"]),
                            "questions": row.fillna("")["Pending Questions"],
                            "assumptions": row.fillna("")["Assumptions"]} for i, row in
            df.iterrows()}


def beautifyBool(foo, warning=True):
    """
    :type foo: bool
    :param warning: If True, returns warning symbol instead of red X if foo is False
    :type warning: bool

    :return: emoji based on bool, green check mark if True

    Requires the `emoji` to return emojis.
    """
    try:
        import emoji
        emojize = lambda x: emoji.emojize(x, use_aliases=True)
    except ModuleNotFoundError:
        return "O" if foo else "X"
    bad = emojize(":warning:") if warning else emojize(":x:")
    return emojize(":white_check_mark:") if foo else bad
