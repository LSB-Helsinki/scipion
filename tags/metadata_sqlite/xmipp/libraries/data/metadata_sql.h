/***************************************************************************
 *
 * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

#ifndef METADATASQL_H
#define METADATASQL_H

#include <iostream>
#include <map>
#include "strings.h"
#include <external/sqlite-3.6.23/sqlite3.h>
#include "metadata_label.h"
#include "metadata.h"


class MDSqlStaticInit;
class MDQuery;
class MetaData;

/** MDSql this class will manage SQL database interactions
 * will be used to implement different MetaData functions
 * The class will be static(only static methods)
 * and only available for MetaData
 */
class MDSql
{
public:

    /**Return an unique id for each metadata
     * this function should be called once for each
     * metada and the id will be used for operations
     */
    static int getMdUniqueId();

    /**This will create the table to store
     * the metada objects, will return false
     * if the mdId table is already present
     */
    static bool createMd(const MetaData *mdPtr);

    /**This function will drop the entire table
     * for use the metada again, a call
     * to 'createMd' should be done
     */
    static bool clearMd(const MetaData *mdPtr);

    /**This function will drop the entire table
     * for use the metada again, a call
     * to 'createMd' should be done
     */
    static bool copyMd(const MetaData *mdPtrIn, MetaData *mdPtrOut);

    /**Add a new row and return the objId(rowId)
     *
     */
    static long int addRow(const MetaData *mdPtr);

    /** Add a new column to a metadata
     */
    static bool addColumn(const MetaData *mdPtr, MDLabel column);
    /**Set the value of an object in an specified column
     *
     */
    static bool setObjectValue(const MetaData *mdPtr, const int objId, const MDLabel column, const MDValue &value);

    /**Get the value of an object
     *
     */
    static bool getObjectValue(const MetaData *mdPtr, const int objId, const MDLabel column, MDValue &value);

    /** This function will do a select
     * the 'limit' is the maximum number of object
     * returned, if is -1, all will be returned
     * Also a query could be specified for selecting objects
     * if no query is provided by default all are returned
     */
    static std::vector<long int> selectObjects(const MetaData *mdPtr, int limit = -1, const MDQuery *queryPtr = NULL);

    /** This function will delete elements
     * that match the query
     * if not query is provided, all rows are deleted
     */
    static long int deleteObjects(const MetaData *mdPtr, const MDQuery *queryPtr);

    /** Coppy the objects from a metada to other
     * return the number of objects copied
     * */
    static long int copyObjects(const MetaData *mdPtrIn, MetaData *mdPtrOut, const MDQuery *queryPtr = NULL);

    static char *errmsg;
    static const char *zLeftover;
    static int rc;

private:
    static int table_counter;
    static sqlite3 *db;

    static MDSqlStaticInit initialization; //Just for initialization

    ///Just call this function once, at static initialization
    static bool sqlBegin();
    static bool sqlEnd();
    static bool sqlCommit();
    static bool dropTable(const int mdId);
    static bool createTable(const int mdId, const std::vector<MDLabel> * labelsVector = NULL);
    static bool insertValues(double a, double b);
    static bool execSingleStmt(const std::string &stmtStr);
    static std::string tableName(const int tableId);

    friend class MDSqlStaticInit;
}
;//close class MDSql

class MDSqlStaticInit
{
private:
    MDSqlStaticInit()
    {
        MDSql::sqlBegin();
    }//close constructor

    ~MDSqlStaticInit()
    {
        MDSql::sqlEnd();
    }//close destructor

    friend class MDSql;
}
;//close class MDSqlStaticInit


///@defgroup MetaDataQuery represent queries to a metadata
//@ingroup DataLibrary

/** MDQuery this is the base class for queries, its abstract
 *@ingroup MetaDataQuery
 */
class MDQuery
{
public:

    /**This now is specific to the SQL implementation
     * and its requiered to all MDQuery subclasses
     * equal to 0 means that is ABSTRACT in this class
     * and only accesible for MetaData
     */
    std::string queryString;
    int limit;
};

/**MDValueEqual this will test if a label have a value
 *@ingroup MetaDataQuery
 */
class MDValueEqual: public MDQuery
{
public:
    MDLabel label;

    template <class T>
    MDValueEqual(MDLabel label, const T &value)
    {
        std::stringstream ss;
        MDValue mdValue;
        MDL::voidPtr2Value(label, (void*)new T(value), mdValue);
        ss << MDL::label2Str(label) << "=";
        MDL::value2Stream(label, mdValue, ss);
        this->queryString = ss.str();
    }
};//class MDValueEqual

/**MDValueEqual this will test if a label have a value
 *@ingroup MetaDataQuery
 */
class MDValueRange: public MDQuery
{
public:
    template <class T>
    MDValueRange(MDLabel label, const T &valueMin, const T &valueMax)
    {
        std::stringstream ss;
        MDValue mdValue;
        MDL::voidPtr2Value(label, (void*)new T(valueMin), mdValue);
        ss << MDL::label2Str(label) << ">=";
        MDL::value2Stream(label, mdValue, ss);
        ss << " AND ";
        MDL::voidPtr2Value(label, (void*)new T(valueMax), mdValue);
        ss << MDL::label2Str(label) << "<=";
        MDL::value2Stream(label, mdValue, ss);
        this->queryString = ss.str();

    }
};//class MDValueRange

#endif
