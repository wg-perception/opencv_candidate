/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef CREATIVE_H_
#define CREATIVE_H_

#include <boost/thread.hpp>

#include <opencv2/core/core.hpp>

#include <DepthSense.hxx>

namespace creative
{
  class Reader
  {
  public:
    Reader()
    {
    }

    ~Reader()
    {
      context_.quit();
    }

    /** Start the capture thread. This can be called beforehand to make sure data is synchronized
     * as color is started and the depth node is registered some time after (otherwise, everything crashes)
     */
    static void
    initialize();

    /** Returns the status of the reader
     * @return
     */
    static bool
    isInitialized()
    {
      return is_initialized_;
    }
    /** Return the current images
     * @param color
     * @param depth
     */
    static void
    getImages(cv::Mat&color, cv::Mat& depth);
  private:
    static void
    run();
    static void
    onNewColorSample(DepthSense::ColorNode obj, DepthSense::ColorNode::NewSampleReceivedData data);
    static void
    onNewDepthSample(DepthSense::DepthNode obj, DepthSense::DepthNode::NewSampleReceivedData data);
    static bool is_initialized_;
    static DepthSense::Context context_;
    static boost::thread thread_;
    static DepthSense::ColorNode color_node_;
    static DepthSense::DepthNode depth_node_;
    static cv::Mat color_;
    static cv::Mat depth_;
    static boost::mutex mutex_;
  };
}

#endif /* CREATIVE_H_ */
