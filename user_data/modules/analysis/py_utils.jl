import TimeZones.TimeZone, TimeZones.ZonedDateTime

@doc raw"""
    dtseries_to_zones(series)

Convert a pandas series to a ZonedDateTime array

"""
function dtseries_to_zoned(series)
    return ZonedDateTime.(series, TimeZone(series.dt.tz.zone))
end
